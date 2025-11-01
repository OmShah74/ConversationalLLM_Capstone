"""
SmolLM2 Knowledge Distillation Script
Distills SmolLM2-360M to a smaller pruned model
"""
import itertools
from datasets import concatenate_datasets, IterableDataset, Dataset
import yaml
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from itertools import islice
import argparse
import os

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def pad_logits(student_logits, teacher_logits):
    """Pads logits to match vocabulary size between student and teacher."""
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)

    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)

        if student_size < teacher_size:
            pad_tensor = torch.zeros(
                *student_logits.shape[:-1], pad_size, 
                dtype=student_logits.dtype, device=student_logits.device
            )
            return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits)
        else:
            pad_tensor = torch.zeros(
                *teacher_logits.shape[:-1], pad_size, 
                dtype=teacher_logits.dtype, device=teacher_logits.device
            )
            return (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
            
    return student_logits, teacher_logits

def load_models_and_tokenizers(config):
    """Loads student and teacher models and tokenizers."""
    model_kwargs = {"torch_dtype": torch.bfloat16}

    if config.get("model_config", {}).get("use_flash_attention"):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    print(f"Loading models with kwargs: {model_kwargs}") 

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["teacher"], 
        trust_remote_code=True
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )

    # Set padding token if not present
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
        print(f"Set student pad_token to eos_token: {student_tokenizer.pad_token}")

    print(f"Loading teacher model: {config['models']['teacher']}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["teacher"], 
        trust_remote_code=True,
        **model_kwargs
    )
    
    print(f"Loading student model: {config['models']['student']}")
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True,
        **model_kwargs
    )
    
    print(f"Teacher model parameters: {teacher_model.num_parameters():,}")
    print(f"Student model parameters: {student_model.num_parameters():,}")
    
    return teacher_model, student_model, teacher_tokenizer, student_tokenizer

# ====================================================================
# CUSTOM SCHEDULER
# ====================================================================

class CustomSchedulerCallback(TrainerCallback):
    """Callback to implement custom learning rate scheduler with warmup and annealing."""
    
    def __init__(self, warmup_steps, total_steps, initial_phase_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_phase_steps = initial_phase_steps
        self.scheduler = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        optimizer = kwargs['optimizer']
        
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < self.warmup_steps:
                return current_step / max(1, self.warmup_steps)
            # Constant learning rate phase
            elif current_step < self.initial_phase_steps:
                return 1.0
            # Linear annealing phase
            else:
                return max(0.0, 1.0 - (
                    (current_step - self.initial_phase_steps) 
                    / max(1, self.total_steps - self.initial_phase_steps)
                ))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Custom scheduler initialized:")
        print(f"  - Warmup steps: {self.warmup_steps}")
        print(f"  - Constant phase until step: {self.initial_phase_steps}")
        print(f"  - Total steps: {self.total_steps}")
        
    def on_step_end(self, args, state, control, **kwargs):
        if self.scheduler is not None:
            self.scheduler.step()

# ====================================================================
# DISTILLATION TRAINER
# ====================================================================

class DistillationTrainer(SFTTrainer):
    """Custom trainer for knowledge distillation."""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config', None)
        self.teacher_model = kwargs.pop('teacher_model', None)
        super().__init__(*args, **kwargs)

        # Move teacher to same device as student
        if self.teacher_model.device != self.model.device:
            self.teacher_model = self.teacher_model.to(self.model.device)
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        print(f"Teacher model on device: {self.teacher_model.device}")
        print(f"Student model on device: {self.model.device}")
        
    def pad_logits(self, *args):
        return pad_logits(*args)

    def forward_kl_divergence(self, student_logits, teacher_logits):
        """Compute KL divergence loss between student and teacher."""
        temperature = self.config["distillation"]["temperature"]

        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        kl_div = F.kl_div(
            student_log_probs, 
            teacher_probs,
            reduction='batchmean',
            log_target=False
        )
        
        # Scale by temperature squared
        return kl_div * (temperature ** 2)

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute combined distillation loss."""
        # Pad logits if vocabulary sizes differ
        student_logits, teacher_logits = self.pad_logits(student_logits, teacher_logits)
        
        # KL divergence loss
        kl_loss = self.forward_kl_divergence(student_logits, teacher_logits)
        
        alpha = self.config["distillation"]["alpha"]
        
        # If alpha < 1, also compute cross-entropy loss
        if alpha < 1.0: 
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1),
                ignore_index=-100
            )
            combined_loss = alpha * kl_loss + (1 - alpha) * ce_loss
        else:
            combined_loss = kl_loss

        return combined_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss for a batch."""
        # Get device
        if hasattr(model, 'module'): 
            device = model.module.device 
        else:
            device = next(model.parameters()).device 

        # Move inputs to device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()} 

        # Forward pass through student
        student_outputs = model(**inputs) 

        # Forward pass through teacher (no gradients)
        with torch.no_grad(): 
            teacher_outputs = self.teacher_model(**inputs) 

        # Get labels
        labels = inputs.get('labels', inputs.get('input_ids')) 

        if labels is None: 
            raise ValueError("Neither 'labels' nor 'input_ids' found in inputs.")

        # Compute distillation loss
        loss = self.distillation_loss(
            student_outputs.logits, 
            teacher_outputs.logits, 
            labels
        )
        
        return (loss, student_outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only, 
                       ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation loop to properly compute distillation loss."""
        
        eval_loss = 0.0
        num_examples = 0
        
        # Call parent to get metrics structure
        output = super().evaluation_loop(
            dataloader, description, True, ignore_keys, metric_key_prefix
        )

        # Compute actual distillation loss
        self.model.eval()
        with torch.no_grad():
            for inputs in dataloader:
                loss = self.compute_loss(self.model, inputs)
                batch_size = inputs["input_ids"].size(0)
                eval_loss += loss.item() * batch_size
                num_examples += batch_size

        # Update metrics with correct loss
        if num_examples > 0:
            eval_loss /= num_examples
            output.metrics[f"{metric_key_prefix}_loss"] = eval_loss
        
        return output
    
# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main(config_path):
    print("="*60)
    print("SmolLM2 Knowledge Distillation")
    print("="*60)
    
    # 1. LOAD CONFIGURATION
    config = load_config(config_path)
    print(f"\nLoaded config from: {config_path}")
    
    # 2. LOAD MODELS AND TOKENIZERS
    print("\n" + "="*60)
    print("Loading Models and Tokenizers")
    print("="*60)
    
    teacher_model, student_model, teacher_tokenizer, student_tokenizer = (
        load_models_and_tokenizers(config)
    )

    # 3. DATA PREPARATION
    print("\n" + "="*60)
    print("Preparing Datasets")
    print("="*60)
    
    # dataset_name = config["dataset"]["name"]
    # print(f"Loading dataset: {dataset_name}")
    
    
    # # Load dataset
    # dataset = load_dataset(
    #     dataset_name,
    #     split=config["dataset"]["split"],
    #     streaming=config["dataset"].get("streaming", True)
    # )

    # Put these imports at top if not already present


    # --- REPLACE the original simple load_dataset(...) block with this ---
    dataset_name = config["dataset"]["name"]
    streaming = config["dataset"].get("streaming", True)
    eval_samples = config["dataset"]["eval_samples"]

    print(f"Loading dataset: {dataset_name} (streaming={streaming})")

    # Case A: User provided multiple subsets in config["dataset"]["subsets"]
    if "subsets" in config["dataset"] and config["dataset"]["subsets"]:
        subset_configs = config["dataset"]["subsets"]

        if streaming:
            # STREAMING: chain the subset iterables together
            subset_iterables = []
            for s in subset_configs:
                cfg_name = s["name"]
                cfg_split = s.get("split", config["dataset"].get("split", "train"))
                print(f" - adding streaming subset: {cfg_name} (split={cfg_split})")
                ds_iter = load_dataset(dataset_name, cfg_name, split=cfg_split, streaming=True)
                subset_iterables.append(ds_iter)

            # Chain them in sequence: first all examples from subset0, then subset1, ...
            chained_iter = itertools.chain(*subset_iterables)

            # Consume a few items to detect columns and build evaluation set
            # We'll take up to min(1000, eval_samples) as a small buffer to inspect structure
            probe = list(itertools.islice(chained_iter, min(1000, eval_samples)))
            if len(probe) == 0:
                raise RuntimeError("No examples found in provided streaming subsets.")

            # Ensure 'text' field present in examples
            if "text" not in probe[0]:
                # print available keys for debugging
                raise ValueError(f"'text' column not found in streamed examples. Example keys: {list(probe[0].keys())}")

            # Rebuild an iterator that yields the probe items first then the remainder
            chained_iter = itertools.chain(iter(probe), chained_iter)

            # Build evaluation list by consuming eval_samples items
            eval_list = list(itertools.islice(chained_iter, eval_samples))
            if len(eval_list) < eval_samples:
                print(f"Warning: requested {eval_samples} eval samples but only got {len(eval_list)}.")

            # Create an IterableDataset-like object for trainer: many trainers accept Python iterators as train_dataset,
            # but better to wrap in datasets.IterableDataset.from_generator if you need an actual IterableDataset object.
            # Here we leave train_dataset as the chained iterator (trainer must support iterable datasets).
            train_dataset = chained_iter
            eval_dataset = Dataset.from_dict({"text": [ex["text"] for ex in eval_list]})

        else:
            # NON-STREAMING: download each subset and concatenate into one Dataset
            ds_list = []
            for s in subset_configs:
                cfg_name = s["name"]
                cfg_split = s.get("split", config["dataset"].get("split", "train"))
                print(f" - loading subset (non-streaming): {cfg_name} (split={cfg_split})")
                ds = load_dataset(dataset_name, cfg_name, split=cfg_split, streaming=False)  # returns a Dataset
                # If ds is a DatasetDict, pick "train" or the provided split; here split param should yield a Dataset
                ds_list.append(ds)

            if len(ds_list) == 0:
                raise RuntimeError("No subsets loaded.")
            if len(ds_list) == 1:
                full_ds = ds_list[0]
            else:
                full_ds = concatenate_datasets(ds_list)

            if "text" not in full_ds.column_names:
                raise ValueError(f"'text' column not found. Available columns: {full_ds.column_names}")

            # Create eval and train by slicing (this materializes only what you slice)
            eval_dataset = full_ds.select(range(min(eval_samples, len(full_ds))))
            total_train_samples = config["dataset"]["total_train_samples"]
            # compute remaining training range (ensure bounds)
            start = min(len(full_ds), len(eval_dataset))
            end = min(len(full_ds), start + max(0, total_train_samples - len(eval_dataset)))
            train_dataset = full_ds.select(range(start, end))

    else:
        # Case B: legacy single-config usage
        split = config["dataset"].get("split", "train")
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)

        if streaming:
            # For streaming single-config, sample eval_samples using islice
            iterator = iter(dataset)
            eval_list = list(itertools.islice(iterator, eval_samples))
            if len(eval_list) == 0:
                raise RuntimeError("No examples found when streaming the dataset.")
            if "text" not in eval_list[0]:
                raise ValueError(f"'text' column not found in dataset. Example keys: {list(eval_list[0].keys())}")
            eval_dataset = Dataset.from_dict({"text": [ex["text"] for ex in eval_list]})
            train_dataset = iterator  # remaining streaming iterator
        else:
            # Non-streaming single config
            if "text" not in dataset.column_names:
                raise ValueError(f"'text' column not found in dataset. Available columns: {dataset.column_names}")
            eval_dataset = dataset.select(range(min(eval_samples, len(dataset))))
            total_train_samples = config["dataset"]["total_train_samples"]
            start = len(eval_dataset)
            end = min(len(dataset), start + max(0, total_train_samples - len(eval_dataset)))
            train_dataset = dataset.select(range(start, end))

    print("Dataset preparation complete.")
    print(f"  - Eval dataset size: {len(eval_dataset)}")
    # Note: train_dataset might be an iterator in streaming mode. Ensure Trainer supports the given type.
    # If trainer requires a datasets.Dataset, you must use non-streaming mode or materialize the train dataset.

    
    # Check if 'text' column exists
    if 'text' not in dataset.column_names:
        raise ValueError(f"'text' column not found in dataset. Available columns: {dataset.column_names}")
    
    eval_samples = config["dataset"]["eval_samples"]
    
    # Create evaluation dataset
    print(f"Creating evaluation dataset with {eval_samples} samples...")
    eval_list = list(islice(dataset, eval_samples))
    eval_dataset = Dataset.from_dict({"text": [ex["text"] for ex in eval_list]})
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # Create training dataset (skip eval samples)
    print(f"Creating training dataset...")
    train_dataset = dataset.skip(eval_samples)

    # 4. CALCULATE TRAINING PARAMETERS
    print("\n" + "="*60)
    print("Calculating Training Parameters")
    print("="*60)
    
    total_samples = config["dataset"]["total_train_samples"] - eval_samples
    batch_size = config["training"]["per_device_train_batch_size"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_epochs = config["training_aux"]["num_train_epochs"]

    effective_batch_size = batch_size * grad_accum_steps * num_gpus
    max_steps = int((total_samples / effective_batch_size) * num_epochs)
    max_steps = max(1, max_steps)

    initial_phase_fraction = config["training_aux"]["annealing_phase_fraction"]
    initial_phase_steps = int(max_steps * (1 - initial_phase_fraction))
    
    save_steps = max(1, int(max_steps * config["training_aux"]["save_steps_fraction"]))
    logging_steps = max(1, int(max_steps * config["training_aux"]["logging_steps_fraction"]))
    eval_steps = max(1, int(max_steps * config["training_aux"]["eval_steps_fraction"]))
    
    warmup_ratio = config["training"].get("warmup_ratio", 0.01)
    warmup_steps = int(max_steps * warmup_ratio)
    
    print(f"Total training samples: {total_samples:,}")
    print(f"Per-device batch size: {batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Max steps: {max_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Annealing starts at step: {initial_phase_steps}")
    print(f"Save every: {save_steps} steps")
    print(f"Log every: {logging_steps} steps")
    print(f"Eval every: {eval_steps} steps")
    
    lr = config["training"]["learning_rate"]
    student_name = config["models"]["student"].split("/")[-1]
    run_name = f'smollm2_distill_{student_name}_lr_{lr}'

    # 5. DEFINE TRAINING ARGUMENTS
    print("\n" + "="*60)
    print("Setting up Training Arguments")
    print("="*60)
    
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        weight_decay=config["training"]["weight_decay"],
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_strategy="steps",
        eval_strategy="steps",
        bf16=config["training"].get("bf16", True),
        fp16=config["training"].get("fp16", False),
        optim="adamw_torch",
        run_name=run_name,
        logging_dir=f"./logs/{run_name}",
        report_to=config["training"].get("report_to", "tensorboard"),
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False,
        lr_scheduler_type="constant",  # We use custom scheduler
        save_total_limit=3,  # Keep only last 3 checkpoints
    )

    # 6. ENABLE GRADIENT CHECKPOINTING IF NEEDED
    if config["training"].get("gradient_checkpointing", False):
        print("Enabling gradient checkpointing...")
        student_model.gradient_checkpointing_enable()

    # 7. INITIALIZE TRAINER
    print("\n" + "="*60)
    print("Initializing Distillation Trainer")
    print("="*60)
    
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=student_tokenizer,
        config=config,
        dataset_text_field="text",
        max_seq_length=config["tokenizer"]["max_length"],
        packing=config["training"].get("packing", True),
    )

    # 8. ADD CUSTOM SCHEDULER CALLBACK
    scheduler_callback = CustomSchedulerCallback(
        warmup_steps=warmup_steps,
        total_steps=max_steps,
        initial_phase_steps=initial_phase_steps
    )
    trainer.add_callback(scheduler_callback)

    # 9. TRAINING EXECUTION
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    try:
        resume_from = config["training"].get("resume_from_checkpoint")
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # 10. SAVE MODEL
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    
    output_dir = config['training']['output_dir']
    print(f"Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    student_tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("Training Completed Successfully!")
    print("="*60)
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for SmolLM2"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    main(args.config)