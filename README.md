# Democratizing-LLM-FT  
### Scalable SFT Workflows Across IDRIS Computing Clusters
---

## 🚀 Practical Large-Scale LLM Fine-Tuning on IDRIS Clusters

We study a realistic Instruct Fine-Tuning (SFT) workflow on **64 GPUs**, corresponding to nearly the full **DALIA** system (16/18 nodes) and a resource scale still achievable on the **Jean Zay** partitions. Models are loaded directly from the **Hugging Face Hub**, reflecting common academic and industrial practices.

At this scale, heavyweight frameworks such as *NeMo-Megatron*, *DeepSpeed*, *Nanotron*, or *TorchTitan* introduce unnecessary complexity because they rely on model-parallel strategies that are not needed for **64-GPU training**. Instead, a **PyTorch-native pipeline**—using HF `transformers/datasets`, **FSDP2**, **selective activation checkpointing**, and **torch.compile**—offers the best combination of flexibility, efficiency, and portability, provided GPUs have **≥ 80 GB** of memory and are connected with a **high-bandwidth interconnect**.

We also evaluate NVIDIA's `HFAutoModelForCausalLLM`, which supports HF model loading but is now deprecated in favor of the newer **NeMo AutoModel**, itself still under development and limited to **NVIDIA-only environments**.

Based on our measurements, a **3,000 H100 GPU-hour allocation** is sufficient to fine-tune large dense models such as **Qwen2.5-72B** in this configuration with `tulu-3-sft-mixture` dataset .

Future work will extend this democratization to **Mixture-of-Experts training** (expert parallelism), **large-context models** (context parallelism), and more advanced stages such as **pre-training** and **post-training with RL**, while tracking progress in **TorchTitan** and **NeMo AutoModel**.

Our realistic Instruct Fine-Tuning scenario follows the setup described in [Tulu 3: Pushing Frontiers in Open Language Model Post-Training (2024)](https://arxiv.org/abs/2411.15124).

Here is the [detailed code review](doc/CODE_REVIEW.md).


## ✅ Conclusion

In a realistic SFT scenario using **small batches (~128 sequences)** on **64 GPUs**, with dense LLMs up to **72B parameters** and a **4096 context length**, and with pretrained weights loaded from the **Hugging Face Hub**, we find that the **FSDP2 + selective activation checkpointing + `torch.compile`** pipeline offers the **best balance of performance, clarity, flexibility, and portability**.

This conclusion holds **only if GPUs provide ≥ 80 GB of memory** and the system offers a **modern high-bandwidth interconnect**. Under these conditions, the workflow remains simple to configure and highly reproducible across heterogeneous HPC environments.  
**This conclusion applies to SFT workloads only—pre-training requires multi-dimensional parallelism (TP/PP/FSDP).**


## ⚠️ Important Note on Scope and Fairness

In this specific scenario — **dense SFT on 64 GPUs**, using pretrained weights from the HF Hub and relying on moderate batch sizes — a **PyTorch-native workflow with Hugging Face Transformers** is the most practical and efficient solution for us. This does **not** place NVIDIA NeMo at a disadvantage intentionally: the framework is simply **not evaluated here in the conditions where it shines**. We fully acknowledge that **NeMo becomes indispensable for other use cases**, such as large-scale pre-training, advanced model-parallelism, Mixture-of-Experts architectures, or complex multi-node pipelines.

Similarly, **TorchTitan** is rapidly gaining relevance and deserves dedicated analysis in future work. Our focus on PyTorch+Transformers in this report reflects the narrow scope of this benchmark, not a general preference in all contexts.

## 📊 Figure: SFT Benchmarking Across IDRIS GPU Clusters

![results](doc/images/SFTBench_results.png)

All evaluations in this study were conducted on **64 GPUs**, with a **global batch size of 128**, a **4096-token sequence length**, and **bf16-mixed precision** (bf16 for compute, FP32 for optimizer states and parameter replicas).

The only hyperparameters that were tuned are:

- For Full pytorch solution, **the selective activation checkpointing ratio**, adjusted according to the available GPU memory
|    sAC ratio    | A100-80GB | H100-80GB | GB200-186GB |
|-----------------|-----------|-----------|-------------|
|  Qwen2.5-14B    | 1/2       | 1/2       | 0           |
|  Qwen2.5-32B    | 3/4       | 3/4       | 0           |
|  Qwen2.5-72B    | 1         | 1         | 1/2         |


- For NeMo solution, **The tensor parallelism dimension**, chosen based on the number of GPUs per compute node

The tensor parallelism dimension is set to **TP=4** on the H100 and DALIA/GB200 partitions (4 GPUs per node) and to **TP=8** on the A100 octo-GPU partition, except for **Qwen2.5-72B**, where we use **TP=4** due to GPU memory limits that prevent TP=8 from running with a gradient accumulation of 1.


### Impact of Interconnect Performance
A strong dependency on the cluster interconnect is visible:
- Moving from **A100-OmniPath** to **H100-InfiniBand** yields a **×10 improvement** in throughput, mainly due to OmniPath becoming the limiting factor in distributed training.
- Going from **H100-InfiniBand** to **B200-NVLink** provides an additional **×2 speed-up**, thanks to full-node NVLink enabling much higher FSDP2 throughput.

### PyTorch FSDP2 + sAC + `torch.compile`
On **H100-InfiniBand** and **B200-NVLink**, our simple PyTorch pipeline (**FSDP2 + selective Activation Checkpointing + `torch.compile`**) delivers the best performance among all tested configurations.  
These results highlight that:
- selective AC reduces overhead compared to full AC  
- `torch.compile` contributes a significant boost across all model sizes  
- FSDP2 scales efficiently when the interconnect is not the bottleneck  

### NeMo HFAutoModelForCausalLLM (TP + FSDP2)

On the older **A100-OmniPath** partition, the best-performing configuration is the **Tensor Parallelism (TP) + FSDP2** setup from NeMo. In this scenario, a **2D parallelism strategy** is employed across the 64 GPUs:  
- **Tensor Parallelism** spans *all GPUs inside each node* (TP size = 4 or 8 depending on the hardware),  
- while **FSDP2** shards *across nodes*, with an effective FSDP group size of **16 or 8 GPUs**.

This hybrid layout is better suited to a **low-bandwidth interconnect** like OmniPath, since TP keeps most communication *intra-node* (high throughput), while FSDP only synchronizes across a smaller cross-node group. This makes the approach more efficient for very large models and helps compensate for the inter-node bottleneck where FSDP alone would be critical.

### Hollow Bars — Cross-Checking the Two Baseline Configurations

The **hollow bars** are included to validate the comparison between the two **solid-bar configurations**, which represent distinct training strategies. These hollow bars correspond to simplified or baseline versions of each approach and should therefore appear **nearly equivalent**:

- The **orange hollow bar** represents **PyTorch FSDP2 + selective AC (sAC) without `torch.compile`**.  
- The **green hollow bar** represents **NeMo HFAutoModelForCausalLLM with full FSDP2 (no TP) + Full Activation Checkpointing + Liger Kernels**.

The small difference between the two baselines is explained primarily by:
- the absence of **selective** activation checkpointing on the NeMo baseline (NeMo only enables **full AC**),  
- and the potential impact of **Liger Kernels** on memory layout and compute behavior.

**GPU Memory Usage Notes** : The NeMo baseline (green hollow bar) generally shows **higher GPU memory usage** than the PyTorch baseline (orange hollow bar).

Despite these discrepancies, the two hollow baselines are close enough to serve as useful reference points for interpreting the differences shown by the corresponding solid-bar configurations.

### Notes
**Training time estimation:** we measured the **average iteration duration over 100 steps** (excluding the first) and multiplied this value by the total number of training steps.  
**Important:** the hyperparameters shown in the code **must not be considered as reference**, because the **gradient descent was not tuned or monitored** — this was a **benchmark-only setup**, not an optimized training run.

## Containers vs. Modules — Practical Observations

For portability and reproducibility across heterogeneous systems, we chose to rely on an NGC container image (`nemo-25.09`) rather than `module` or virtual/conda environments. In practice, the container setup proved **significantly more performant** than the `module` environment for this heavily distributed workload, particularly due to more consistent CUDA/NCCL integration.

However, containers can also hide networking issues. We strongly recommend enabling NCCL diagnostics using `NCCL_DEBUG=WARN`, as we observed cases where the interconnect failed to detect its intended interfaces and silently fell back to a **degraded communication mode**. Monitoring these warnings is essential to ensure that distributed performance remains optimal.

## ⚡ Improved Efficiency with 32 GPUs on DALIA (GB200-NVL72)

Thanks to the **GB200-NVL72 nodes** on DALIA — each offering **192 GB of GPU RAM** — it becomes possible to run all three models (14B, 32B, 72B) on **only 32 GPUs** using the full PyTorch pipeline (FSDP2 + selective activation checkpointing + `torch.compile`).  
With a **per-GPU batch size of 4**, we reach a **global batch size (GBS) of 128**.

Under these conditions, the total GPU-hours improve significantly for the 14B model :

- **Qwen-14B:** 4.64 h → **148.48 GPU-hours** (vs. 234.88 GPU-hours)  
- **Qwen-32B:** 13.43 h → **429.76 GPU-hours** (vs. 469.12 GPU-hours)  
- **Qwen-72B:** 35.27 h → **1128.6 GPU-hours** (vs. 1142.4 GPU-hours)  


## 🔁 Reproducing the Experiments

To reproduce our benchmarks, follow the steps below:

### 1. Adjust local paths
Modify the paths in the training scripts (`./*.py`) so that:
- model checkpoints,
- dataset locations,
- and output (logging, checkpoints) folders match your local filesystem or shared cluster environment.

### 2. Configure your SLURM environment
If you are running on a SLURM-managed cluster, adapt the provided SLURM job files inside `slurm/`:
- account/project name  
- partition or QoS  
- number of nodes / GPUs per node  
- container or module settings  
- job duration and output folders  

### 3. Launch the experiments
Once the SLURM scripts are adjusted, you can run the experiments directly:

```bash
sbatch slurm/FSDP_sAC_h100_72B.slurm
sbatch slurm/DALIA_NeMo_FSDP_TP_32B.slurm

```

## Poster
![poster](doc/images/Posterv1.png)

----


## 📎 Attribution

This work is openly available to the community.  
If you reuse our scripts, methodology, or benchmark results, please cite or acknowledge this project.  
It supports open, transparent, and reproducible LLM research on HPC infrastructures.



