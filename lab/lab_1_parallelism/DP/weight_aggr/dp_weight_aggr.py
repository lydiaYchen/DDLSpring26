import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.distributed as dist
# get our dataset
from simplellm.dataloaders import TinyStories
# get our models
from simplellm.llama import CausalLLama, LLama
# our loss
from simplellm.losses import causalLLMLoss
# get our tokenizer
from simplellm.tokenizers import SPTokenizer
from torch.optim import AdamW

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
script_dir = Path(__file__).parent
dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 1
device = "cuda"


def worker(rank: int, world_size: int) -> None:
    os.chdir(script_dir)

    with open(f"out{rank}.txt", "w", buffering=1) as f, redirect_stdout(f):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.manual_seed(0)

        # make the tokenizer
        tokenizer = SPTokenizer()
        # make the model
        net = LLama(CausalLLama, tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                    device=device, n_layers=n_layers, ctx_size=seq_l, padding_idx=tokenizer.pad_id)
        # skip so we can have different things
        ds = TinyStories(tokenizer, batch_size=batch_size,
                         seq_l=seq_l, skip=rank*5000)
        # we can iterate the dataset with:
        iter_ds = iter(ds)

        optim = AdamW(net.parameters(), lr=8e-4)

        sizes = []
        len_sizes = []
        for param in net.parameters():
            sizes.append(param.shape)
            len_sizes.append(len(param.view(-1)))

        for itr in range(5_000):
            optim.zero_grad()
            x = next(iter_ds)
            target = x.clone().detach()
            x = x.to(device)

            x = net(x)
            loss = causalLLMLoss(x, target, tokenizer.vocab_size)
            # log the loss
            print(itr, loss.item())
            loss.backward()
            optim.step()
            torch.cuda.empty_cache()

            # wait for everyone
            dist.barrier()
            tmp = []

            for param in net.parameters():
                if param == None:
                    tmp.append(torch.zeros_like(param).view(-1))
                    continue

                tmp.append(param.view(-1))
                param.grad = None

            prev_grad = torch.cat(tmp).to("cpu")
            dist.all_reduce(prev_grad, op=dist.ReduceOp.SUM)
            tmp = torch.split(prev_grad, len_sizes)

            with torch.no_grad():
                for i, param in enumerate(net.parameters()):
                    # average
                    param = tmp[i].view(sizes[i]).to(device) / world_size


def main() -> None:
    world_size = 3
    ctx = multiprocessing.get_context("spawn")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=world_size, mp_context=ctx) as executor:
        executor.map(worker, range(world_size), [world_size] * world_size)

    print(f"Elapsed time (s): {(time.time() - start_time):.2f}")


if __name__ == "__main__":
    main()
