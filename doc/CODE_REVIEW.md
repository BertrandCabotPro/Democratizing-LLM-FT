# Code review

## Selective Activation Checkpointing (sAC)

Activation checkpointing is essential to reduce the memory footprint during LLM training.  
Instead of storing all intermediate activations during the forward pass, PyTorch replays part of the computation during the backward pass, keeping only what is strictly necessary.

In our implementation, **selective activation checkpointing (sAC)** allows applying checkpointing only on a *fraction* of the Transformer blocks, giving fine-grained control over the trade-off between:
- GPU memory usage  
- Runtime overhead   

The **runtime overhead introduced** by sAC depends directly on the selected ratio and typically ranges from 0% (no checkpointing) up to **~20%** under full activation checkpointing

## 🔧 Enabling Selective Activation Checkpointing

```python
### Selective Activation Checkpointing
if args.sac:
    model.config.use_cache = False
    BlockCls = type(model.model.layers[0])
    apply_fsdp_checkpointing(model, BlockCls, args.sac)
```


## Gradient Accumulation (Last-Resort Memory Relief)

When GPU memory becomes fully saturated — even with **full activation checkpointing** enabled — the only remaining option is to reduce the per-GPU batch size and compensate using **gradient accumulation**. This technique splits a large batch into several micro-batches processed sequentially, accumulating gradients before applying an optimizer step. While it effectively lowers memory usage, its drawback is a **runtime penalty that is almost linear** in this context: using `grad_acc=2` nearly doubles the iteration time, and so on. Gradient accumulation should therefore be considered a **last-resort solution** when all other memory-optimization strategies have been exhausted.

## Collate Function for Instruct Fine-Tuning

In Instruct Fine-Tuning with dialogue-style datasets, the `collate_function` must correctly prepare inputs and labels for causal language modeling. Tokens that belong to the *non-assistant role part* (e.g., user or system messages) must be assigned the label **`-100`**, which lies outside the vocabulary range and is therefore ignored by the Cross-Entropy loss. Only the padding tokens are masked as well, ensuring that the model learns exclusively from the assistant’s response tokens.

For benchmarking purposes, our pipeline pads (or truncates) **all sequences to a fixed `max_seq_length`**, ensuring a constant computational shape across training steps. In standard training practice, however, sequences are padded only up to the **maximum length within each batch** and truncated at `max_seq_length`, offering better memory and runtime efficiency.


## FSDP2 with Mixed Precision (BF16)

We rely on **PyTorch FSDP2** to shard model parameters and optimizer states across GPUs while using **mixed precision** to balance numerical stability and performance. In our setup, parameters are stored in `float32` in their sharded form, but exposed as `bfloat16` when unsharded for compute. This follows the design described in the [official PyTorch FSDP2 tutorial.](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

Below is a minimal example of how we configure **FSDP2 in BF16-mixed mode**:

```python
fsdp_kwargs = {
    "mp_policy": MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
}

for layer in model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)

# sharded parameters are float32
for param in model.parameters():
    assert param.dtype == torch.float32

# unsharded parameters are bfloat16
model.unshard()
for param in model.parameters(recurse=False):
    assert param.dtype == torch.bfloat16
model.reshard()
```

## Loading the Model from the Hugging Face Hub

Models are loaded from the **Hugging Face Hub** using the standard API:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="bfloat16",
)
```

For large models in the tens of billions of parameters, memory pressure appears at two stages:

1. **On GPU**: a full BF16 or FP32 copy can easily exceed device memory if parameters are not sharded early.

2. **On CPU**: naïvely casting the whole model to float32 or loading directly in float32 causes the entire parameter set to reside in host RAM at once, which frequently leads to CPU OOM.

The `.from_pretrained` method is **relatively CPU-memory efficient** when loading the model in its **original precision** (often BF16), because it streams weights without duplicating them unnecessarily. However, FSDP2 mixed precision expects parameters to be managed internally in **float32** (with compute typically in BF16), which creates a tension:

* **Casting on CPU** → large, temporary FP32 copy in host RAM → high risk of OOM
* **Casting on GPU** → large FP32 footprint before sharding → high risk of OOM as well

To resolve this, we cast to FP32 while sharding, layer by layer:

```python
for layer in model.model.layers:
    fully_shard(layer.type(torch.float32), **fsdp_kwargs)
fully_shard(model.type(torch.float32), **fsdp_kwargs)
```
This pattern preserves **CPU and GPU memory** by avoiding a global FP32 copy of the model in RAM. The trade-off is a **non-negligible casting time overhead** (on the order of 100 to 1000 seconds, depending on model size and hardware), but it keeps the loading process feasible for very large models without hitting CPU or GPU out-of-memory errors.


## FSDP2 Model Checkpointing — Saving & Loading with Distributed Checkpoint (DCP)

PyTorch FSDP2 introduces a new, robust mechanism for saving and loading model weights using the **Distributed Checkpoint (DCP)** APIs.  
These APIs allow exporting a **full, consolidated state dict** from a sharded FSDP2 model, while minimizing CPU memory usage thanks to options such as `cpu_offload=True` and `mmap=True`.

---

### Saving a Full FSDP2 State Dict

With `get_model_state_dict`, PyTorch reassembles the full model state from all shards.  
The `cpu_offload=True` option ensures that reconstruction happens safely even for large models.

```python
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

model_state_dict = get_model_state_dict(
    model=model,
    options=StateDictOptions(
        full_state_dict=True,   # consolidate all shards into a single full state dict
        cpu_offload=True,       # avoids holding the full model in GPU memory
    )
)

torch.save(model_state_dict, "model_state_dict.pt")
```
This produces a **single-file checkpoint** compatible with standard `torch.load`, HF tooling, and any downstream environment.

### Loading a Full FSDP2 State Dict

To reload a saved model, we use `set_model_state_dict`.
The `mmap=True` option helps limit CPU memory usage by memory-mapping the checkpoint file instead of loading it entirely into RAM.

```python
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

# mmap=True reduces CPU memory pressure for large models
full_sd = torch.load(
    "checkpoints/model_state_dict.pt",
    mmap=True,
    weights_only=True,
    map_location='cpu',
)

set_model_state_dict(
    model=model,
    model_state_dict=full_sd,
    options=StateDictOptions(
        full_state_dict=True,        # indicates that this is a consolidated checkpoint
        broadcast_from_rank0=True,   # rank 0 loads, other ranks receive via broadcast
    ),
)
```
This properly rehydrates the sharded model across all devices.

### Optimization Tip: Avoid Recasting & Re-Sharding on Every Run

As described earlier, casting the model to **FP32** and **sharding it layer-by-layer** with FSDP2 can take 100–1000 seconds, especially for 30B–70B models.
To avoid this overhead for each experiment starting from the same pretrained weights and the same distributed/sharding configuration, you can:
1. Load the pretrained model from the HF Hub in BF16
2. Cast + shard layer-by-layer with FSDP2 (memory-safe approach)
3. **Save this already-prepared model using DCP (the section above)**
4. Reload it instantly for future experiments

This workflow reduces startup time from **minutes** to **seconds**, avoids repeated casting overhead, and ensures you always start from the exact pretrained weights.

###  FSDP2 Optimizer Checkpointing — Saving & Loading with Distributed Checkpoint (DCP)
In FSDP2, **optimizer states are sharded across GPUs**, just like model parameters.  
The Distributed Checkpoint (DCP) APIs make it possible to save and reload these sharded optimizer states efficiently, without requiring full consolidation on a single machine.
Refer to [pytorch/examples](https://github.com/pytorch/examples/blob/main/distributed/FSDP2/checkpoint.py) for loading and saving optimizer state dicts with `set_optimizer_state_dict` and `get_optimizer_state_dict`.