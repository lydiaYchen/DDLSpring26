
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from commons.utils import get_device
from datasets import load_dataset
from lab_5_decentralized_rl.base import (SYSTEM_PROMPT, Experience, GRPOConfig,
                                         advantage_compute, extract_gsm8k,
                                         gather, generate_rollouts,
                                         grpo_train_loop, reward_answer_binary,
                                         sequences_log_probs)
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
script_dir = Path(__file__).parent
device = get_device()
lr = 5e-6
kl_weight = 0
# completions (rollouts) per prompt
# (divided among workers in horizontal setting)
group_size = 2
# prompts included per step
# (divided among workers in vertical setting)
batch_size = 2
microbatch_size = 1


def worker(rank: int, world_size: int, model_name: str) -> None:
    assert (batch_size % world_size == 0)
    os.chdir(script_dir)

    with open(f"out{rank}.txt", "w", buffering=1) as f, redirect_stdout(f):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.manual_seed(0)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_dataset = load_dataset(
            "openai/gsm8k", "main", split="train", streaming=True)
        prompt_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=False
        )
        data_interp = extract_gsm8k
        reward_func = reward_answer_binary
        grpo_config = GRPOConfig(num_generations=group_size,
                                 micro_batch_size=microbatch_size)

        for k, prompt_batch in enumerate(prompt_loader):
            replay_buffer = []
            rollout_returns = []
            rollout_indv = []
            questions, _solutions, answers = data_interp(prompt_batch)
            generation_times = 0
            comm_times = 0

            with (torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16)):
                idx = -1
                for q, a in zip(questions, answers):
                    idx += 1

                    if idx % world_size != rank:
                        continue

                    gen_start = time.time()
                    prompt_ids, prompt_mask, completion_ids, completion_mask = generate_rollouts(
                        model=model, tokenizer=tokenizer, question=q,
                        sys_prompt=SYSTEM_PROMPT, num_rollouts=group_size)
                    completions = tokenizer.batch_decode(
                        completion_ids, skip_special_tokens=True)
                    returns, _, _ = reward_func(completions, a)

                    if len(replay_buffer) == 0:
                        print(completions[0])

                    sequence_ids = torch.cat(
                        (prompt_ids, completion_ids), dim=1)
                    attention_mask = torch.cat(
                        (prompt_mask, completion_mask), dim=1)
                    seq_log_probs, _ = sequences_log_probs(
                        model, sequence_ids=sequence_ids, attention_mask=attention_mask,
                        logits_to_keep=completion_ids.size(1)
                    )
                    generation_times += time.time() - gen_start
                    rollout_indv.append(returns)
                    returns = returns.to(device)
                    comm_start = time.time()
                    seq_log_probs = gather(seq_log_probs, world_size)
                    attention_mask = gather(
                        attention_mask, world_size)
                    completion_ids = gather(
                        completion_ids, world_size)
                    returns = gather(returns, world_size)
                    completion_mask = gather(
                        completion_mask, world_size)
                    sequence_ids = gather(sequence_ids, world_size)
                    comm_times += time.time() - comm_start

                    for loc_rank in range(world_size):
                        advantages = advantage_compute(returns[loc_rank])
                        rollout_returns.append(returns[loc_rank].to("cpu"))
                        exp = Experience(
                            sequence_ids=sequence_ids[loc_rank],
                            advantages=advantages,
                            attention_mask=attention_mask[loc_rank],
                            action_mask=completion_mask[loc_rank],
                            start_ids=0,
                            logits_to_keep=completion_ids[loc_rank].size(1),
                            gen_log_probs=seq_log_probs[loc_rank])
                        replay_buffer.append(exp.to("cpu"))

            print(f"generation time of step {k}: {generation_times:.4f}")
            print(f"communication time of step {k}: {comm_times:.4f}")
            torch.cuda.empty_cache()
            episode_reward = torch.stack(rollout_returns).mean()
            print(f"group returns of step {k}: {episode_reward:.4f}")
            episode_reward = torch.stack(rollout_indv).mean()
            print(f"idividual returns of step {k}: {episode_reward:.4f}")
            torch.cuda.empty_cache()
            update_start = time.time()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss_hist, kl_hist, entropy_hist = grpo_train_loop(
                    model, optimizer, replay_buffer, grpo_config)
                print(f"Loss at step {k}: {loss_hist[0]}")
                print(f"KL loss at step {k}: {kl_hist[0]}")
                print(f"Entropy at step {k}: {entropy_hist[0]}")

            print(f"update time of step {k}: {time.time() - update_start}")

        model.save_pretrained("./outputs/")


def main() -> None:
    world_size = 2
    ctx = multiprocessing.get_context("spawn")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=world_size, mp_context=ctx) as executor:
        list(executor.map(
            worker, range(world_size), [world_size] * world_size,
            ["HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]))

    print(f"Elapsed time (s): {(time.time() - start_time):.2f}")


if __name__ == "__main__":
    main()
