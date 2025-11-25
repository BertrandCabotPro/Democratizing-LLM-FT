# Democratizing-LLM-FT - Scalable SFT Workflows Across IDRIS Computing Clusters
----------
### Practical Large-Scale LLM Fine-Tuning on IDRIS Clusters
We consider a realistic Instruct Fine-Tuning scenario using 64 GPUs, corresponding to the full DALIA system and a level of resource still accessible on the Jean Zay GPU partitions. Pretrained weights are loaded directly from the Hugging Face Hub, reflecting standard academic and industrial workflows. In this setup, heavy frameworks such as NeMo-Megatron, DeepSpeed, Nanotron, or TorchTitan add unnecessary complexity at this scale, relying on model-parallel strategies that are not required for **64-GPU training**. A **PyTorch-native pipeline**—using HF transformers/datasets, FSDP2, selective activation checkpointing, and torch.compile—offers the most flexible and efficient solution, assuming a modern interconnect and high-memory GPUs. We also compare it with *NeMo’s HFAutoModelForCausalLLM*, which allows loading HF models but but is now deprecated in favor of the newer NeMo AutoModel, itself still under active development, and restricted to NVIDIA GPU environments.
 
Access to IDRIS’s GPU clusters enables **simple, efficient, and reproducible** LLM fine-tuning using these PyTorch techniques, even under realistic HPC constraints. Based on our measurements, a **3,000 H100 GPU-hour allocation** is sufficient to complete a full Instruct Fine-Tuning run in this configuration (e.g. Qwen2.5-72B). Future work will extend this democratization to **Mixture-of-Experts** (expert parallelism), **large-context training** (context parallelism), and more advanced stages such as pre-training and RL-based post-training, while monitoring ongoing progress in **TorchTitan** and **NeMo AutoModel**.


## SFT Benchmarking Results

![results](doc/images/SFTBench_results.png)

## ✅ Conclusion

In a realistic Instruct Fine-Tuning scenario using **small batch sizes (~128 sequences per step)** across a limited surface of **64 GPUs**, with dense LLMs up to **72B parameters** (no Mixture-of-Experts) and a **4096 context length**, and assuming pretrained weights loaded directly from the **Hugging Face Hub**, we conclude that the **PyTorch FSDP2 + selective activation checkpointing + `torch.compile`** workflow offers the **best balance of performance, flexibility, clarity, and portability**.

This conclusion holds **only when GPUs provide sufficient memory (≥ 80 GB)** and are connected through a **high-bandwidth interconnect**. Under these conditions, the approach remains fully open, easy to configure, and deployable across heterogeneous systems, making it the most practical and robust solution for large-scale SFT workloads within this resource envelope. **This conclusion applies to SFT workloads, not pre-training, where multi-dimensional parallelism becomes mandatory.**

In this context, adding **tensor parallelism** increases operational complexity without delivering meaningful benefits. Introducing **pipeline parallelism** makes the workflow even more complex, as it requires redefining the model architecture and injecting pretrained weights across multiple shards. By contrast, **FSDP/FSDP2 handles sharding transparently**, making large-scale training feel almost seamless. However, FSDP alone becomes limiting when scaling to **very large GPU counts** driven by extreme model sizes or full **pre-training workloads**, where more advanced parallelism strategies may become necessary.


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

## 📦 Loading the Model from the Hugging Face Hub

Models are loaded from the **Hugging Face Hub** using the standard API:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="bfloat16",
)

For large models in the tens of billions of parameters, memory pressure appears at two stages:

1. On GPU: a full BF16 or FP32 copy can easily exceed device memory if parameters are not sharded early.

2. On CPU: naïvely casting the whole model to float32 or loading directly in float32 causes the entire parameter set to reside in host RAM at once, which frequently leads to CPU OOM.

The .from_pretrained method is relatively CPU-memory efficient when loading the model in its original precision (often BF16), because it streams weights without duplicating them unnecessarily. However, FSDP2 mixed precision expects parameters to be managed internally in float32 (with compute typically in BF16), which creates a tension:

Casting on CPU → large, temporary FP32 copy in host RAM → high risk of OOM

Casting on GPU → large FP32 footprint before sharding → high risk of OOM as well

To resolve this, we cast to FP32 while sharding, layer by layer:

```python
for layer in model.model.layers:
    fully_shard(layer.type(torch.float32), **fsdp_kwargs)
fully_shard(model.type(torch.float32), **fsdp_kwargs)
```
This pattern preserves **CPU and GPU memory** by avoiding a global FP32 copy of the model in RAM. The trade-off is a **non-negligible casting time overhead** (on the order of 100 to 1000 seconds, depending on model size and hardware), but it keeps the loading process feasible for very large models without hitting CPU or GPU out-of-memory errors.


## Points d'intérêts et de discussion
* selective Activation Checkpointing
* Gradient Accumulation
* FSDP2 implementation
* Instruct Fine Tuning collate function
* Model Loading & precision
* Model Checkpointing
* Container usage vs `module load`


## Relancer les expériences
`sbatch slurm/machin.slurm`


## Poster
![poster](doc/images/Poster3%20-%20nov25(3).png)


