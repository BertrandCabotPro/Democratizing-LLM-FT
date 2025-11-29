import os
import datasets
import functools

import time
import torch
import torch.distributed as dist
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from pathlib import Path
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.aggregation import RunningMean
from torchmetrics.text import Perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils import (
    make_sft_collate,
    apply_fsdp_checkpointing,
    Chronometer,
    memory_usage
)

# Distribution Variables
RANK = int(os.environ['SLURM_PROCID'])
LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
## Don't forget to export in slurm file 
#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
#export MASTER_PORT=29500


if RANK == 0:
    print(f">>> Training on {WORLD_SIZE} processes")
    print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']} MASTER_PORT: {os.environ['MASTER_PORT']}")

def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Training related arguments
    parser.add_argument('--bsz', "--batch-size", dest="batch_size", default=1, type=int, help='batch size per GPU')
    parser.add_argument('--seq-len', "--seq-length", dest="seq_length", default=4096, type=int, help='sequence length of each sample per GPU')
    parser.add_argument('--grad-acc', default=1, type=int, help='Gradient Accumulation count')
    parser.add_argument('--epochs', default=2, type=int, help='Number of epochs')

    # Benchmarking
    parser.add_argument('--test', default=False, action='store_true', help='Test 100 iterations')
    parser.add_argument('--test-nsteps', default=100, type=int, help='the number of steps in test mode')
    parser.add_argument("--optimizer-precision", default=False, action=BooleanOptionalAction, help="whether or not to print precision of optimizer states item.")
    parser.add_argument("--cpu-usage", default=False, action=BooleanOptionalAction, help="whether or not to print CPU memory Usage.")

    # JIT related arguments
    parser.add_argument("--compile", default=False, action=BooleanOptionalAction, help="whether or not to compile model")

    # DataLoader related arguments
    parser.add_argument('--num-workers', default=4, type=int, help='num workers in dataloader')
    parser.add_argument('--persistent-workers', default=True, action=BooleanOptionalAction, help='activate persistent workers in dataloader')
    parser.add_argument('--pin-memory', default=True, action=BooleanOptionalAction, help='activate pin memory option in dataloader')
    parser.add_argument('--non-blocking', default=True, action=BooleanOptionalAction, help='activate asynchronuous GPU transfer')
    parser.add_argument('--prefetch-factor', default=3, type=int, help='prefectch factor in dataloader')
    parser.add_argument('--drop-last', default=False, action=BooleanOptionalAction, help='activate drop_last option in dataloader')

    # Optimizer AdamW related arguments
    parser.add_argument("--lr-warmup-ratio", default=0.1, type=float, help="linear warmup of learning rate before cosine annealing")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=1e-5, help="learning rate for adamw")
    parser.add_argument("--wd", "--weight-decay", dest="weight_decay", type=float, default=0.1, help="weight decay for adamw")

    # Other
    parser.add_argument("--model", default='Qwen/Qwen2.5-32B-Instruct', type=str, help="HuggingFaceHub Model's Name")
    parser.add_argument("--selective-activation-checkpointing", "--sac",
                        dest="sac",
                        default=None,
                        type=str,
                        help='For a given ac ratio p, we should essentially apply ac on every "1/p" blocks.')
    
    return parser.parse_args()


args = parse_args()
os.environ["TOKENIZER_PARALLELISM"] = "false"

chrono = Chronometer(RANK, args.grad_acc)

## Distribution initialization
torch.cuda.set_device(LOCAL_RANK)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dist.init_process_group(
    init_method="env://", #Default value
    backend="nccl",
    rank=RANK,
    world_size=WORLD_SIZE,
)

gbs=args.batch_size*args.grad_acc*WORLD_SIZE
if RANK == 0: print(
    f"world size:{WORLD_SIZE}, GBS:{gbs}, BSperDev:{args.batch_size}, seq. length:{args.seq_length}, sAC ratio:{args.sac}, grad accumulation:{args.grad_acc}, compile:{args.compile}"
    )


#####Path to define
### Jean Zay
DSDIR = Path(os.environ["DSDIR"])
model_path = DSDIR / "HuggingFace_Models" / args.model
dataset_path = "/lustre/fswork/dataset/tulu-3-sft-mixture/data"
### DALIA
#DSDIR = Path(os.environ.get("ALL_WORK", "")) / "BC"
#model_path = DSDIR / "HuggingFace_Models" / args.model
#dataset_path = DSDIR /  "tulu-3-sft-mixture/data"
############


#### Initialize the model and its tokenizer
if RANK == 0: chrono.tac_time(clear=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16", trust_remote_code=True)
num_parameters = sum(param.numel() for param in model.parameters())
tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side="left")
if RANK == 0: print(f"Time to load and initialize the model and its tokenizer: {chrono.tac_time():.3f}s")
####

### Selective Activation Checkpointing
if args.sac:
    model.config.use_cache = False
    BlockCls = type(model.model.layers[0])
    apply_fsdp_checkpointing(model, BlockCls, args.sac)

#### Distribute the Model
if RANK == 0: chrono.tac_time(clear=True)
    
fsdp_kwargs = {}
fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

for layer in model.model.layers:
    fully_shard(layer.type(torch.float32), **fsdp_kwargs)
fully_shard(model.type(torch.float32), **fsdp_kwargs)

if RANK == 0: print(f"Time to shard the model: {chrono.tac_time():.3f}s")

# Transfer to  GPU
model = model.to(device, non_blocking=args.non_blocking)

if RANK == 0: print(f"Time to transfer the model to GPU: {chrono.tac_time():.3f}s")

#### JIT
if args.compile:
    model = torch.compile(model)

    if RANK == 0: print(f"Time to instantiate torch.compile: {chrono.tac_time():.3f}s")
####

if RANK == 0:
    #print(f"model: {model}")
    print(f"number of parameters: {num_parameters}")
    print(f'Pre-loop Model MaxMemory for GPU:{RANK} {torch.cuda.max_memory_allocated()/2**30} GBytes')


#### Data Loading
train_dataset = datasets.load_dataset("parquet", data_files=str(dataset_path) + '/*.parquet', split="train")  # 
collate_fn = make_sft_collate(tokenizer, max_seq_length=args.seq_length)

sampler = DistributedSampler(
    dataset=train_dataset,
    rank=RANK,
    num_replicas=WORLD_SIZE,
    shuffle=True,
)

dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=args.pin_memory,
    drop_last=args.drop_last,
    persistent_workers=args.persistent_workers,
    prefetch_factor=args.prefetch_factor,
    sampler=sampler,
)
####

if RANK == 0: print(f"Time to load dataset and initialize dataloader: {chrono.tac_time():.3f}s")

#### Training step
criterion = CrossEntropyLoss(ignore_index=-100)
optimizer = AdamW(
    params=model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    eps=1e-05,
)

if RANK == 0:
    print(f'global batch size: {args.batch_size * WORLD_SIZE} - mini batch size: {args.batch_size}')
    print(f"DATALOADER {args.num_workers} {args.persistent_workers} {args.pin_memory} {args.non_blocking} {args.prefetch_factor} {args.drop_last} ")
    print(f"Optimizer: {optimizer}")

lr_warmup_iters = int(len(dataloader) * args.lr_warmup_ratio)  * args.epochs / args.grad_acc
warmup_lr_scheduler = LinearLR(
    optimizer,
    start_factor=1e-9,
    end_factor=1,
    total_iters=lr_warmup_iters,
)
annealing_lr_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=len(dataloader) * args.epochs / args.grad_acc - lr_warmup_iters,
    eta_min=0.,
)
lr_scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_lr_scheduler,annealing_lr_scheduler],
    milestones=[lr_warmup_iters]
)

loss_metric = RunningMean(window=5).to(device)
perplexity = Perplexity(ignore_index=-100).to(device)
####


#### Training loop
if args.test:
    chrono.start()
    chrono.dataload()
    
if RANK == 0: chrono.tac_time(clear=True)

if args.test: args.epochs = 1 #Test with only 100 steps
for epoch in range(args.epochs):
    #set epoch for sampler
    sampler.set_epoch(epoch)
    for i, (input_ids, attention_mask, labels) in enumerate(dataloader, start=1):
        if args.test and i > args.test_nsteps * args.grad_acc: break
    
        input_ids = input_ids.to(device, non_blocking=args.non_blocking)
        attention_mask = attention_mask.to(device, non_blocking=args.non_blocking)
        labels = labels.to(device, non_blocking=args.non_blocking)
    
        if args.test:
            chrono.dataload()
            chrono.training()
            chrono.forward()
    
        # passes and weights update
        logits: torch.Tensor = model(input_ids, attention_mask=attention_mask).logits
        bsz, seq_len, vocab_size = logits.shape
        loss: torch.Tensor = criterion(logits.view(bsz * seq_len, vocab_size), labels.view(bsz * seq_len))
        loss /= WORLD_SIZE
        loss /= args.grad_acc
        
        
        loss_metric.update(loss)
        perplexity.update(logits, labels)
        
        if args.test: 
            chrono.forward()
            chrono.backward()
            
        loss.backward()
        
        if i % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        if args.test:
            chrono.backward()
            chrono.training()
        
        step = (i // args.grad_acc) + 1
        if step % 10 == 0 and i % args.grad_acc == 0:
            L = loss_metric.compute()
            perp = perplexity.compute()
            last_lr = lr_scheduler.get_last_lr()[0]
            if RANK == 0:
                print(f"Step {step} / {args.test_nsteps if args.test else len(dataloader) // args.grad_acc} | Loss: {L.item():.3f} | Perplexity: {perp.item():.3f} | LR: {last_lr:0.3e} | Wall: {chrono.tac_time():.3f}")
    
        if args.test: chrono.dataload()

        if i==1 and RANK == 0: print(f"Time to first step - compile Graph Building: {chrono.tac_time():.3f}s")
####

    ######### Model Checkpointing at each epoch ############
    
    if not args.test:
        ckeckpoint_name =  f"model_state_dict_{os.environ['SLURM_JOB_ID']}_epoch{epoch}.pt"
        print(f"Model Checkpointing - Building the {ckeckpoint_name} file")
        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
        torch.save(model_state_dict, ckeckpoint_name)



    ###############################

if args.test: chrono.display()

dist.barrier()
if RANK == 0:
    print(f'MaxMemory for GPU:{RANK} {torch.cuda.max_memory_allocated()/2**30} GBytes')

if args.optimizer_precision and RANK == 0:
    for k, v in optimizer.state.items():
        print(k, {kk: vv.dtype for kk, vv in v.items()})
        break

if args.cpu_usage and RANK == 0:
    print(f"RANK: {RANK}")
    memory_usage()



dist.barrier()
dist.destroy_process_group()
