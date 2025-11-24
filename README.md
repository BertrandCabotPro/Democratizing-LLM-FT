# Democratizing-LLM-FT - Scalable SFT Workflows Across IDRIS Computing Clusters
----------
### Practical Large-Scale LLM Fine-Tuning on IDRIS Clusters
We consider a realistic Instruct Fine-Tuning scenario using 64 GPUs, corresponding to the full DALIA system and a level of resource still accessible on the Jean Zay GPU partitions. Pretrained weights are loaded directly from the Hugging Face Hub, reflecting standard academic and industrial workflows. In this setup, heavy frameworks such as NeMo-Megatron, DeepSpeed, Nanotron, or TorchTitan add unnecessary complexity at this scale, relying on model-parallel strategies that are not required for **64-GPU training**. A **PyTorch-native pipeline**—using HF transformers/datasets, FSDP2, selective activation checkpointing, and torch.compile—offers the most flexible and efficient solution, assuming a modern interconnect and high-memory GPUs. We also compare it with *NeMo’s HFAutoModelForCausalLLM*, which allows loading HF models but remains immature and restricted to NVIDIA-only systems.
 
Access to IDRIS’s GPU clusters enables **simple, efficient, and reproducible** LLM fine-tuning using these PyTorch techniques, even under realistic HPC constraints. Based on our measurements, a **3,000 H100 GPU-hour allocation** is sufficient to complete a full Instruct Fine-Tuning run in this configuration (e.g. Qwen2.5-72B). Future work will extend this democratization to **Mixture-of-Experts** (expert parallelism), **large-context training** (context parallelism), and more advanced stages such as pre-training and RL-based post-training, while monitoring ongoing progress in **TorchTitan** and **NeMo AutoModel**.


## SFT Benchmarking Results

![results](doc/images/SFTBench_results.png)

## ✅ Conclusion

In a realistic Instruct Fine-Tuning scenario using a limited surface of **64 GPUs**, with dense LLMs up to **72B parameters** (no Mixture-of-Experts) and a **4096 context length**, and assuming pretrained weights loaded directly from the **Hugging Face Hub**, we conclude that the **PyTorch FSDP2 + selective activation checkpointing + `torch.compile`** workflow offers the **best balance of performance, flexibility, clarity, and portability**.

This conclusion holds **only when GPUs provide sufficient memory (≥ 80 GB)** and are connected through a **high-bandwidth interconnect**. Under these conditions, the approach remains fully open, easy to configure, and deployable across heterogeneous systems, making it the most practical and robust solution for large-scale SFT workloads within this resource envelope.

In this context, adding **tensor parallelism** increases operational complexity without delivering meaningful benefits. Introducing **pipeline parallelism** makes the workflow even more complex, as it requires redefining the model architecture and injecting pretrained weights across multiple shards. By contrast, **FSDP/FSDP2 handles sharding transparently**, making large-scale training feel almost seamless. However, FSDP alone becomes limiting when scaling to **very large GPU counts** driven by extreme model sizes or full **pre-training workloads**, where more advanced parallelism strategies may become necessary.


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


