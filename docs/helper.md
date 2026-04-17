# DDP and FSDP Memory Calculations for Qwen 2.5 7B

All calculations below use the Qwen 2.5 7B architecture as a running example.

| Constant | Value |
|---|---|
| Parameters | 7 billion |
| Hidden size (h) | 3,584 |
| Intermediate size (i) | 18,944 |
| Layers | 28 |
| Precision (FP16) | 2 bytes per value |

---

## 1. DDP: Activation Memory

For the Qwen 2.5 7B architecture (28 layers, 3,584 hidden dimension) in FP16 (2 bytes per value), a 2 GB activation footprint typically implies a configuration like this on each GPU:

| Parameter | Value |
|---|---|
| Hidden size | 3,584 |
| Sequence length | ~10,000 tokens |
| Batch size (per GPU) | 1 |
| Precision | 2 bytes (FP16/BF16) |
| Layers | 28 |

```
batch_size x bytes x hidden x seq_len x layers
= 1 x 2 x 3,584 x 10,000 x 28
~ 2 GB
```

---

## 2. DDP: Ring AllReduce

The 14 GB gradient tensor (7B params x 2 bytes) is split into 4 equal chunks (A, B, C, D) of 3.5 GB each across 4 GPUs. The ring algorithm runs in two phases.

### Phase 1: Reduce Scatter

The goal is for each GPU to end up with the globally summed version of exactly one chunk.

**Step 1: first handoff**
Each GPU sends a different chunk to its neighbor.

```
G1 sends A1 to G2    (G2 now has A1 + A2)
G2 sends B2 to G3    (G3 now has B2 + B3)
G3 sends C3 to G4    (G4 now has C3 + C4)
G4 sends D4 to G1    (G1 now has D4 + D1)
```
Traffic: 3.5 GB sent per GPU.

**Step 2: adding the second piece**
Each GPU passes the accumulated piece it just received.

```
G1 sends (D4+D1) to G2    (G2 now has D4+D1+D2)
G2 sends (A1+A2) to G3    (G3 now has A1+A2+A3)
G3 sends (B2+B3) to G4    (G4 now has B2+B3+B4)
G4 sends (C3+C4) to G1    (G1 now has C3+C4+C1)
```
Traffic: 3.5 GB sent per GPU.

**Step 3: final sum**
One last move so each GPU has the total sum for its owned chunk.

```
G1 sends (C3+C4+C1) to G2    G2 now owns the total sum of C
G2 sends (D4+D1+D2) to G3    G3 now owns the total sum of D
G3 sends (A1+A2+A3) to G4    G4 now owns the total sum of A
G4 sends (B2+B3+B4) to G1    G1 now owns the total sum of B
```
Traffic: 3.5 GB sent per GPU.

**Phase 1 result:** 10.5 GB sent per GPU. Every GPU now owns the globally summed version of one quarter of the gradient.

### Phase 2: All Gather

The goal is to share those completed sums so every GPU ends up with the full 14 GB of averaged gradients.

Starting state after phase 1:

| GPU | Owns |
|---|---|
| G1 | total sum of B |
| G2 | total sum of C |
| G3 | total sum of D |
| G4 | total sum of A |

**Step 4: first share**

```
G1 sends total B to G2    (G2 now has C + B)
G2 sends total C to G3    (G3 now has D + C)
G3 sends total D to G4    (G4 now has A + D)
G4 sends total A to G1    (G1 now has B + A)
```
Traffic: 3.5 GB sent per GPU.

**Step 5: second share**

```
G1 sends total A (received in step 4) to G2
G2 sends total B (received in step 4) to G3
G3 sends total C (received in step 4) to G4
G4 sends total D (received in step 4) to G1
```
Traffic: 3.5 GB sent per GPU.

**Step 6: final share**

```
G1 sends total D to G2
G2 sends total A to G3
G3 sends total B to G4
G4 sends total C to G1
```
Traffic: 3.5 GB sent per GPU.

**Phase 2 result:** 10.5 GB sent per GPU. Every GPU now has total A, B, C and D. The 14 GB gradient is now identical across the entire 4 node cluster.

### Total Ring AllReduce cost

| Phase | Data sent per GPU |
|---|---|
| Phase 1 (reduce scatter) | 10.5 GB |
| Phase 2 (all gather) | 10.5 GB |
| Total | 21 GB |

By breaking the 14 GB into pieces and moving them like a relay race, each GPU only had to send 21 GB of data to sync a 14 GB model. The general formula is `2 x (N-1)/N x model_size` where N is the number of GPUs.

---

## 3. DDP + QLoRA: LoRA Parameter Calculation (Rank 64)

### Architectural constants

| Constant | Value |
|---|---|
| Hidden size (h) | 3,584 |
| Intermediate size (i) | 18,944 |
| Layers (L) | 28 |
| Rank (r) | 64 |

### Attention projections (Q, K, V, O)

There are 4 projections per layer. Each LoRA adapter adds a down projection (h to r) and an up projection (r to h).

```
r x h x 2 x 4 x L
= 64 x 3,584 x 2 x 4 x 28
= 51,380,224 parameters
```

### MLP projections (gate, up, down)

There are 3 projections per layer. Each involves the hidden size (h) and the intermediate size (i).

```
r x (h + i) x 2 x 3 x L
= 64 x (3,584 + 18,944) x 2 x 3 x 28
= 121,110,528 parameters (total for all 3 projections)
```

Note: for a single MLP projection the count is `64 x (3,584 + 18,944) x 2 x 28 = 40,370,176`.

### Grand total

| Component | Parameters |
|---|---|
| Attention (Q, K, V, O) | 51,380,224 |
| MLP (gate, up, down) | 121,110,528 |
| Total | 172,490,752 (~172.5M) |
| Memory (FP16) | 172.5M x 2 bytes = ~0.34 GB |

---

## 4. Double Quantization in bitsandbytes

Double quantization (also called nested quantization) reduces memory by quantizing the quantization constants themselves.

### The problem

When you quantize a model to 4 bit, it is done in small blocks (typically 64 weights) to maintain precision. Each block requires a scaling factor (usually a 32 bit float) to map the 4 bit numbers back to their original range. While the weights are compressed, these thousands of 32 bit scaling factors create a memory tax of about 0.5 bits per parameter.

### How it works

1. **First quantization:** model weights are compressed into 4 bit NF4
2. **Second quantization:** the 32 bit scaling factors from step 1 are collected into larger blocks (e.g. 256) and quantized again into 8 bit floats

### Impact on memory

| Configuration | Overhead per parameter |
|---|---|
| Standard 4 bit | ~0.5 bits |
| Double quantization | ~0.127 bits |
| Savings | ~0.37 to 0.4 bits |

For a 65B parameter model, this can free up roughly 3 GB of VRAM.

### How to enable it

In `BitsAndBytesConfig`, set `bnb_4bit_use_double_quant=True`. It is highly recommended for constrained hardware as it provides extra memory savings with virtually no loss in model quality.

---

## 5. Adam Optimizer Memory Tax (56 GB)

Even though weights and gradients are in FP16 (2 bytes), the Adam optimizer works in FP32 (4 bytes) to maintain mathematical precision. If it used lower precision, the small updates would get rounded to zero and the model would stop learning.

For every parameter (7 billion of them), Adam tracks two running averages:

### First moment (momentum): 28 GB

A running average of all previous gradients for that specific weight. If a weight has been moving in the same direction for the last 10 steps, momentum keeps it moving that way even if the current gradient is noisy.

```
7B params x 4 bytes = 28 GB
```

### Second moment (variance): 28 GB

A running average of the squared gradients. This tracks how volatile the weight is. If the gradient is jumping around wildly, the optimizer slows down the learning rate for that specific weight to stay stable.

```
7B params x 4 bytes = 28 GB
```

### Total optimizer state

| Component | Memory |
|---|---|
| First moment (momentum) | 28 GB |
| Second moment (variance) | 28 GB |
| Total | 56 GB |

---

## 6. FSDP: Sharding Across dim[0]

`dim[0]` refers to rows (the first dimension of a weight matrix). `dim[1]` refers to columns.

When FSDP shards a weight matrix, it performs a horizontal cut along the row dimension. For a matrix with 4,000 rows across 4 GPUs:

| GPU | Rows |
|---|---|
| GPU 0 | 0 to 999 |
| GPU 1 | 1,000 to 1,999 |
| GPU 2 | 2,000 to 2,999 |
| GPU 3 | 3,000 to 3,999 |

### Per layer math

| Property | Value |
|---|---|
| Total model size (FP16) | 14 GB (7B params x 2 bytes) |
| Number of layers | 28 |
| Average size per layer | 14 / 28 = 0.5 GB |

### Activation size per layer

| Parameter | Value |
|---|---|
| Hidden size | 3,584 |
| Sequence length | ~10,000 tokens |
| Batch size (per GPU) | 1 |
| Precision | 2 bytes (FP16/BF16) |
| Layers | 1 (per layer calculation) |

```
batch_size x bytes x hidden x seq_len x 1
= 1 x 2 x 3,584 x 10,000 x 1
~ 70 MB per layer
```
