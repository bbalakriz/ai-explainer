# DDP vs FSDP: implementation differences explained

This document walks through the real code differences between the DDP (Distributed Data Parallel) and FSDP (Fully Sharded Data Parallel) implementations in this repo. Both train the same QLoRA fine tuned model on the same data, but they distribute the work across GPUs in fundamentally different ways.

## The core idea

DDP copies the entire model to every GPU. Each GPU processes a different batch of data, computes gradients and then all GPUs synchronize their gradients before updating weights. Simple but memory hungry.

FSDP shards (splits) the model parameters, gradients and optimizer states across GPUs. Each GPU only holds a slice of the model at any time. When a layer needs its full parameters for a forward or backward pass, FSDP temporarily gathers them from other GPUs and then releases them. More complex but far more memory efficient.

---

## 1. How the model gets placed on GPUs

### DDP: explicit device map per rank

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map={"": local_rank},   # full model loaded onto this GPU
    torch_dtype=torch.float16,
    trust_remote_code=True
)
```

Each process loads the full model and pins it to its own GPU using `device_map={"": local_rank}`.

### FSDP: no device map, set device first

```python
torch.cuda.set_device(local_rank)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    # no device_map here
    torch_dtype=torch.float16,
    trust_remote_code=True
)
```

FSDP calls `torch.cuda.set_device(local_rank)` before loading and then lets the FSDP framework handle sharding and placement. Providing a `device_map` would conflict with how FSDP manages parameters.

### Why it matters

DDP needs every GPU to hold the full model in VRAM. FSDP only needs each GPU to hold its shard, so larger models can fit on the same hardware.

---

## 2. Quantization config needs an extra field for FSDP

### DDP

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

### FSDP

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.float16,   # extra line for FSDP
)
```

### Why it matters

`bnb_4bit_quant_storage=torch.float16` tells bitsandbytes to store quantized weights as contiguous fp16 tensors. FSDP can only shard regular floating point tensors. Without this, the packed int4 format that bitsandbytes uses by default cannot be split across GPUs.

---

## 3. FSDP sharding and wrapping config in SFTConfig

This is the biggest difference between the two. DDP requires zero extra config because the HF Trainer enables it automatically when `torchrun` launches multiple processes. FSDP needs explicit configuration.

### DDP SFTConfig

No distributed training flags at all. The trainer detects multiple processes and uses DDP by default.

### FSDP SFTConfig (additional lines)

```python
training_args = SFTConfig(
    # ... all the same base args as DDP ...

    fsdp=["full_shard", "auto_wrap"],
    fsdp_config={
        "backward_prefetch": "backward_pre",
        "forward_prefetch": "False",
        "use_orig_params": "False",
        "sync_module_states": "True",
    },
    optim="paged_adamw_32bit",
    ddp_find_unused_parameters=False,
)
```

### What each setting does

| Setting | Purpose |
|---|---|
| `fsdp=["full_shard", "auto_wrap"]` | Enable full sharding of parameters, gradients and optimizer states. Auto wrap model layers into FSDP units. |
| `backward_prefetch: backward_pre` | Start fetching the next shard before the current backward pass finishes so compute and communication overlap. |
| `forward_prefetch: False` | Don't prefetch during forward pass. Saves memory at the cost of some speed. |
| `use_orig_params: False` | Use FSDP's flat parameter representation for better memory efficiency. |
| `sync_module_states: True` | Broadcast the model from rank 0 to all other ranks at startup. Required for QLoRA because only rank 0 loads the quantized model. |
| `optim="paged_adamw_32bit"` | Use a paged optimizer that can spill optimizer states to CPU RAM when GPU memory runs low. |
| `ddp_find_unused_parameters=False` | Disable unused parameter detection. FSDP handles this differently and this flag must be False. |

---

## 4. Gradient accumulation steps

| | DDP | FSDP |
|---|---|---|
| `gradient_accumulation_steps` | 16 | 8 |

FSDP's memory savings from sharding free up enough VRAM that you don't need to accumulate as many micro batches to simulate a large effective batch size. Fewer accumulation steps means fewer forward passes per logical training step.

---

## 5. Extra pip installs for FSDP

The FSDP TrainJob YAML includes an extra dependency upgrade that the DDP version does not need:

```bash
pip install -q -U transformers trl peft accelerate bitsandbytes
```

FSDP + QLoRA support requires newer versions of these libraries (especially `accelerate` and `bitsandbytes`) than what ships in the base training image. DDP works fine with the stock versions.

---

## Quick reference: side by side comparison

| Aspect | DDP | FSDP |
|---|---|---|
| Model placement | `device_map={"": local_rank}` (full copy per GPU) | No device_map, uses `torch.cuda.set_device()` (FSDP shards it) |
| What lives on each GPU | Entire model, all gradients and all optimizer states | Only a shard of parameters, gradients and optimizer states |
| BitsAndBytesConfig | Standard 4 fields | Adds `bnb_4bit_quant_storage=torch.float16` |
| SFTConfig additions | None (DDP is automatic) | `fsdp`, `fsdp_config`, `optim`, `ddp_find_unused_parameters` |
| Gradient accumulation | 16 steps | 8 steps |
| Extra pip installs | No | Yes (newer accelerate and bitsandbytes) |
| Memory usage per GPU | Higher | Lower |
| Config complexity | Minimal | More involved |

---

## When to pick which

Use DDP when the full model (with quantization and LoRA adapters) fits comfortably on each GPU. It is simpler to configure, easier to debug and has less communication overhead.

Use FSDP when the model is too large to fit on a single GPU even with quantization, or when you want to free up GPU memory for larger batch sizes or longer sequence lengths.

Both approaches use the same `torchrun` launcher, the same Kubeflow TrainingRuntime structure and the same training script entrypoint. The differences are entirely in the Python training code and the dependency versions.
