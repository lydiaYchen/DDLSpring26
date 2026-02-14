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
from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage
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
n_layers_glob = 6
seq_l = 256
batch_size = 3
device = "cuda"


def worker(rank: int, world_size: int) -> None:
    os.chdir(script_dir)
    n_layers = n_layers_glob // world_size
    # avoid cuBLAS no context warning
    torch.empty(0, device=device)

    with open(f"out{rank}.txt", "w", buffering=1) as f, redirect_stdout(f):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.manual_seed(0)

        # make the model
        if rank == 0:
            tokenizer = SPTokenizer()
            net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                                  device=device, n_layers=n_layers, ctx_size=seq_l)
            # no skip
            ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
            iter_ds = iter(ds)
        elif rank == 1:
            net = LLamaStage(dmodel=dmodel, num_heads=num_heads,
                             device=device, n_layers=n_layers, ctx_size=seq_l)
        elif rank == 2:
            tokenizer = SPTokenizer()
            net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                                 device=device, n_layers=n_layers, ctx_size=seq_l)
            # no skip
            ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
            iter_ds = iter(ds)

        optim = AdamW(net.parameters(), lr=8e-4)

        for itr in range(5_000):
            optim.zero_grad()

            # FORWARD PASS
            if rank == 0:
                out = next(iter_ds)
                out = out.to(device)
                out = net.embed(out)

                dist.send(out.to("cpu"), 1)
            elif rank == 1:
                inp_batch = torch.empty((batch_size, seq_l, dmodel))
                dist.recv(inp_batch, 0)
                with torch.no_grad():
                    inp_batch = inp_batch.to(device)
                    inp_batch.requires_grad_()
                    inp_batch.retain_grad()

                out = net(inp_batch)
                dist.send(out.to("cpu"), 2)
            elif rank == 2:
                target = next(iter_ds)
                inp_batch = torch.empty((batch_size, seq_l, dmodel))
                dist.recv(inp_batch, 1)
                with torch.no_grad():
                    inp_batch = inp_batch.to(device)
                    inp_batch.requires_grad_()
                    inp_batch.retain_grad()

                logits = net(inp_batch)
                loss = causalLLMLoss(logits, target, tokenizer.vocab_size)
                print(itr, loss.item())
                loss.backward()

            # BACKWARD PASS
            if rank == 2:
                dist.send(inp_batch.grad.to("cpu"), 1)
            elif rank == 1:
                inp_grad = torch.empty((batch_size, seq_l, dmodel))
                dist.recv(inp_grad, 2)
                out.backward(inp_grad.to(device))
                dist.send(inp_batch.grad.to("cpu"), 0)
            elif rank == 0:
                inp_grad = torch.empty((batch_size, seq_l, dmodel))
                dist.recv(inp_grad, 1)
                out.backward(inp_grad.to(device))

            optim.step()
            torch.cuda.empty_cache()


def main() -> None:
    world_size = 3
    ctx = multiprocessing.get_context("spawn")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=world_size, mp_context=ctx) as executor:
        executor.map(worker, range(world_size), [world_size] * world_size)

    print(f"Elapsed time (s): {(time.time() - start_time):.2f}")


if __name__ == "__main__":
    main()
