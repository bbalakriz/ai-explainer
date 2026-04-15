import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import shutil
import os
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig  
from datasets import load_dataset
from datetime import datetime
import json
import mlflow
from transformers import TrainerCallback

class MLflowLineageCallback(TrainerCallback):
    """
    custom callback to inject S3 dataset lineage and LoRA hyperparameters 
    into the MLflow run automatically created by Hugging Face Trainer
    """
    def __init__(self, custom_args, peft_config, train_ds, val_ds, test_ds):
        self.custom_args = custom_args
        self.peft_config = peft_config
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def on_train_begin(self, args, state, control, **kwargs):
        # only execute on the main process (Rank 0) to prevent FSDP duplication
        if state.is_world_process_zero:
            print("Injecting S3 data lineage into Hugging Face's active MLflow run...")
            
            # (note: HF automatically logs learning_rate, batch_size, epochs, etc. 
            # so we omit them here to prevent MLflow "duplicate param" errors)
            mlflow.log_params({
                "model_name": self.custom_args.model_name,
                "data_path": self.custom_args.data_path,
                "lora_r": self.peft_config.r,
                "lora_alpha": self.peft_config.lora_alpha,
                "lora_dropout": self.peft_config.lora_dropout,
            })
            
            # convert datasets and log them directly into HF's run
            ds_train = mlflow.data.from_pandas(
                self.train_ds.to_pandas(), source=self.custom_args.dataset_source, name="intent_train"
            )
            ds_val = mlflow.data.from_pandas(
                self.val_ds.to_pandas(), source=self.custom_args.dataset_source, name="intent_val"
            )
            ds_test = mlflow.data.from_pandas(
                self.test_ds.to_pandas(), source=self.custom_args.dataset_source, name="intent_test"
            )
            
            mlflow.log_input(ds_train, context="training")
            mlflow.log_input(ds_val, context="evaluation")
            mlflow.log_input(ds_test, context="test")

def format_messages_for_training(row: dict, base_model_id: str) -> dict:
    """
    translates the enterprise data schema into a model agnostic chat template for SFTTrainer
    from
    {
        "user_message": "now do this",
        "intent": "ABCD",
        "session_history": [
            {"role": "bot", "message": "something"},
            {"role": "user", "message": "something else"}
        ]
    }

    to 

    {
        "messages": [
            {
            "role": "system",
            "content": "You are an AI intent classifier. Analyze the conversation history and the latest user message. You must output only a valid JSON object containing the predicted 'intent'."
            },
            {
            "role": "assistant",
            "content": "something"
            },
            {
            "role": "user",
            "content": "something else"
            },
            {
            "role": "user",
            "content": "now do this"
            },
            {
            "role": "assistant",
            "content": "{\"intent\": \"ABCD\"}"
            }
        ]
    }
    """
    messages = []
    system_prompt = (
        "You are an AI intent classifier. Analyze the conversation history "
        "and the latest user message. You must output only a valid JSON object "
        "containing the predicted 'intent'."
    )
    
    model_name = base_model_id.lower()
    
    # handle model-specific system prompt support
    if "qwen" in model_name or "phi" in model_name or "deepseek" in model_name:
        messages.append({"role": "system", "content": system_prompt})
    elif "gemma" in model_name:
        pass 
    else:
        messages.append({"role": "system", "content": system_prompt})

    # process session history
    for turn in row.get("session_history", []):
        print (turn)
        hf_role = "assistant" if turn.get("role") == "assistant" else "user"
        messages.append({"role": hf_role, "content": turn.get("content", "")})
        
    # process current message
    current_msg = row.get("user_message", "")
    if "gemma" in model_name and len(messages) == 0:
        current_msg = f"{system_prompt}\n\n{current_msg}"
        
    messages.append({"role": "user", "content": current_msg})
    
    # critical for training: append the target output so the model learns to generate it
    target_output = json.dumps({"intent": row.get("intent", "UNKNOWN")})
    messages.append({
        "role": "assistant",
        "content": target_output
    })
    
    return {"messages": messages}    

def main():
    # parse arguments 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/mnt/data/adapters")
    parser.add_argument("--dataset_source", type=str, default="", help="s3 uri (with versionId) logged to MLflow for data lineage")
    parser.add_argument("--run_name", type=str, default="decoder-fsdp-run-v1", help="mlflow run name, used in both SFTConfig and mlflow.start_run")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0)) # : adding for multi gpu training - to set the device map

    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        if os.path.exists(args.output_dir):
                print(f"cleaning existing output dir to avoid corruption: {args.output_dir}")
                shutil.rmtree(args.output_dir)

    print(f"loading model: {args.model_name}...")

    # qlora config (4-bit loading of the model)
    # this reduces VRAM usage from 16GB -> 6GB, allowed training on T4 16GB GPU (ARO NC64as_T4_v3) 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, # : adding for multi gpu training -  to enable double quant and a lower‑memory compute dtype:
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float16,
    )

    # load base model
    print(f"loading model: {args.model_name,}...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config, 
        torch_dtype=torch.float16, 
        trust_remote_code=True 
    )

    # CRITICAL: Force the base config to fp16 so PEFT doesn't secretly initialize LoRA weights in bfloat16
    model.config.torch_dtype = torch.float16
    model.config.use_cache = False

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # fix for padding issues

    # : Overwrite the Qwen template with a TRL-compatible ChatML template
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n{% generation %}{{ message['content'] }}{% endgeneration %}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )

    # load dataset
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    split_1 = dataset.train_test_split(test_size=0.2, seed=42)
    split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
    train_dataset = split_1["train"]
    val_dataset = split_2["train"]
    test_dataset = split_2["test"]

    if local_rank == 0:
        print("saving quarantined test set to /mnt/data/test_split.jsonl...")
        test_dataset.to_json("/mnt/data/test_split.jsonl")

    # we use lambda to pass the model_name to the mapping function, and remove the old columns
    print("formatting dataset for the specific model...")
    train_dataset = train_dataset.map(
        lambda row: format_messages_for_training(row, args.model_name), 
        remove_columns=["user_message", "intent", "session_history"])
    val_dataset = val_dataset.map(
        lambda row: format_messages_for_training(row, args.model_name), 
        remove_columns=["user_message", "intent", "session_history"])
        # NOTE: needed to tell HF dataset loader that we used these three original columns to build 
        # the 'messages' prompt and we don't need them in RAM anymore
        # so we throw them away so PyTorch doesn't crash

    # define rank (r)
    r = 16 
    peft_config = LoraConfig(
        r=r,       # rank (higher = smarter but slower)
        lora_alpha=(2 * r), # scaling factor for the LoRA updates, typically set to 2x the rank
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, # Increased for stability
        learning_rate=1e-4, # Lowered to prevent mode collapse
        logging_steps=10, # Log more frequently for visibility
        num_train_epochs=8, # Increased for JSON muscle memory
        max_grad_norm=0.3,
        warmup_ratio=0.15, # Gentler ramp-up
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False, # Explicitly deny bfloat16 to prevent T4 crashes
        push_to_hub=False,
        packing=False,
        assistant_only_loss=True, # <-- Critical for intent classification

        # mlflow tracking
        report_to="mlflow",
        run_name=args.run_name,

        # evaluation and checkpointing
        eval_strategy="epoch", # Evaluate at the end of each epoch
        eval_steps=0.1,
        save_strategy="epoch", # Evaluate at the end of each epoch
        save_steps=0.1,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Memory optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,

        fsdp=["full_shard", "auto_wrap"],
        fsdp_config={
            "backward_prefetch": "backward_pre",
            "forward_prefetch": "False",
            "use_orig_params": "False",
            "sync_module_states": "True" # Required for QLoRA + FSDP
        },
        optim="paged_adamw_32bit",
        
        # Keep this! FSDP also requires this to be False
        ddp_find_unused_parameters=False,
    )
    
    lineage_callback = MLflowLineageCallback(
        custom_args=args,
        peft_config=peft_config,
        train_ds=train_dataset,
        val_ds=val_dataset,
        test_ds=test_dataset
    )

    # initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,            # pass the SFTConfig here
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[lineage_callback]
    )

    # train
    print("starting training...")
    trainer.train()

    # save
    output_dir=f"{args.output_dir}/latest" 
    # NOTE: intentionally overriding the run_id with latest for the evaluator to use

    # : multi gpu training - letting the trainer handle FSDP synchronization and saving on its own
    # to avoid the issue of the adapter being saved multiple times by different ranks
    print(f"rank {local_rank}: reached the end of training, calling save_model to save the adapter...")
    if local_rank == 0:
        trainer.save_model(output_dir)
        print(f"adapter saved to: {output_dir}")

        if os.path.exists(output_dir):
            saved_files = os.listdir(output_dir)
            print(f"saved files: {saved_files}")

    # : multi gpu training - not saving the tokenizer on each rank as HF handles it automatically

if __name__ == "__main__":
    main()