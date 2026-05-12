import os
import gc
import re
import math
import numpy as np
from tqdm import tqdm
from typing import Any

import wandb
import torch
import selfies as sf
import random

from molgen.models.dt_gpt import sample
# from molgen.utils.famo import FAMO
from molgen.utils.plot_utils import save_plot


def _as_reward_list(reward_func):
    return reward_func if isinstance(reward_func, list) else [reward_func]


class Trainer:

    def __init__(
            self,
            model,
            train_dataset,
            test_dataset,
            reward_func,
            config,
            save_path=None,
            device: str = "cuda",
            wandb_log=False
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.reward_func = reward_func
        self.save_path = save_path
        self.config = config
        self.bos_token_id = train_dataset.dataset.tokenizer.bos_token_id
        self.eos_token_id = train_dataset.dataset.tokenizer.eos_token_id
        self.pad_token_id = train_dataset.dataset.tokenizer.pad_token_id
        self.ignore_token_id = train_dataset.collate_fn.ignore_index
        self.wandb_log = wandb_log
        self.device = device
        self.optimizer = None
        # Initialize FAMO
        # self.famo = FAMO(
        #     num_tasks=model.config.n_goals,
        #     min_losses=torch.full((model.config.n_goals,), 1e-8, device=device),
        #     lr=config.get("famo_beta", 0.025),
        #     gamma=config.get("famo_gamma", 0.001),
        # )

        # take over whatever gpus are on the system
        # if torch.cuda.is_available():
        #     self.device = torch.cuda.current_device()
        #     self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch, ckpt_name: str = None):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        if ckpt_name is None:
            ckpt_name = f"epoch_{epoch}.pth" if self.test_dataset is None else f"best.pth"

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'token_counter': self.tokens
        }
        torch.save(checkpoint, os.path.join(self.save_path, ckpt_name))

    def load_checkpoint(self, ckpt_name: str = None):
        checkpoint_dir = self.save_path
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint folder {checkpoint_dir} not found, starting training from scratch")
            os.makedirs(checkpoint_dir, exist_ok=True)
            return 0, 0

        if ckpt_name is None:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")]
            if len(checkpoint_files) == 0:
                print(f"No checkpoints to load, starting training from scratch")
                return 0, 0

            epoch_numbers = [
                int(re.search(r"epoch_(\d+)", file).group(1))
                for file in checkpoint_files
                if re.search(r"epoch_(\d+)", file)
            ]
            ckpt_name = f"epoch_{max(epoch_numbers)}.pth"

        else:
            ckpt_name = ckpt_name

        path = os.path.join(checkpoint_dir, ckpt_name)
        checkpoint = torch.load(path)

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tokens = checkpoint['token_counter']

        epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming training from epoch {epoch}")

        return epoch, tokens

    def train(self, optimizer):
        model, config = self.model, self.config
        self.optimizer = optimizer
        gradient_accumulation_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))
        if self.config.get("load_checkpoint", False):
            epoch_n, token_n = self.load_checkpoint(ckpt_name="latest.pth")
        else:
            epoch_n, token_n = 0, 0

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_dataset if is_train else self.test_dataset

            total_loss = 0.0
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                tokens_since_last_step = 0
            num_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                if "cuda" in self.device:
                    batch = {k: v.pin_memory().to(self.device, non_blocking=True) for k, v in batch.items()}
                else:
                    batch = {k: v.to(self.device) for k, v in batch.items()}    # place data on the correct device
                x = batch["input_ids"]  # states
                y = batch["labels"]     # actions
                r = batch["rtgs"]       # rtgs (reward-to-go)
                a = batch["attention_mask"]
                g = batch["goal"]

                # Keep all goals; do not reduce to a single goal during warm-up.
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(
                        input_ids=x, labels=y, targets=y, rtgs=r, attention_mask=a, goal=g
                    )

                loss_value = float(loss.detach().item())
                total_loss += loss_value
                if is_train:
                    # Scale each micro-step by the actual accumulation window size
                    # (handles final partial windows when len(loader) % grad_accum != 0).
                    window_start = (it // gradient_accumulation_steps) * gradient_accumulation_steps
                    window_end = min(window_start + gradient_accumulation_steps, num_batches)
                    effective_accum_steps = window_end - window_start
                    (loss / effective_accum_steps).backward()
                    tokens_since_last_step += int((y != self.ignore_token_id).sum().item())

                    is_accum_boundary = (it + 1) % gradient_accumulation_steps == 0
                    is_last_batch = (it + 1) == len(loader)
                    should_step = is_accum_boundary or is_last_batch
                    if should_step:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        # Decay the learning rate only when optimizer step is performed.
                        if config.get("decay_lr", True):
                            self.tokens += tokens_since_last_step
                            warmup_tokens = config.get("warmup_steps", 0)
                            if self.tokens < warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - warmup_tokens) / float(
                                    max(1, config.get("lr_decay_steps", warmup_tokens * 300) - warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config["learning_rate"] * lr_mult
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr
                        tokens_since_last_step = 0

                    lr = self.optimizer.param_groups[0]['lr']

                    # report progress
                    pbar.set_description(f"epoch {epoch_num + 1} of {epochs} | iter {it}: train loss {loss_value:.5f}. lr {lr:e}")
                    torch.cuda.empty_cache()

            # if not is_train:
            episode_loss = total_loss / len(loader)
            print(f"\nMean Epoch Loss: {episode_loss:.4f}")
            return episode_loss

        best_loss = float('inf')
        best_return = -float('inf')

        self.tokens = token_n  # counter used for learning rate decay
        epochs = config["max_steps"] // len(self.train_dataset)
        epoch_losses = []
        test_loss = best_loss
        reward_eval_every = max(1, int(config.get("reward_eval_every", config.get("eval_every", 1))))
        reward_eval_target = float(config.get("reward_eval_target", 1.0))
        reward_eval_samples = max(1, int(config.get("reward_eval_samples", 10)))
        for epoch in range(epoch_n, epochs):

            epoch_loss = run_epoch('train', epoch_num=epoch)
            epoch_losses.append(epoch_loss)
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            reward_metrics: dict[str, float] = {}
            if (epoch + 1) % reward_eval_every == 0:
                reward_metrics = self.get_returns(ret=reward_eval_target, k=reward_eval_samples)

            if self.wandb_log:
                log_metrics: dict[str, float | int] = {
                    "epoch": epoch,
                    "training_loss": epoch_loss,
                }
                if self.test_dataset is not None:
                    log_metrics["test_loss"] = float(test_loss)
                log_metrics.update(reward_metrics)
                wandb.log(log_metrics)

            # supports early stopping based on the test loss, or save every X epochs if no test set
            good_model = (epoch > 2 and (self.test_dataset is None and (epoch % 1 == 0))) or test_loss < best_loss
            if self.save_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint(epoch)

            # self.save_checkpoint(epoch, ckpt_name="latest.pth")

            # -- pass in target returns
            # model_type = self.model.module.model_type if hasattr(self.model, "module") else self.model.model_type
            # if model_type == 'naive':
            #     eval_return = self.get_returns(0)
            # elif model_type == 'reward_conditioned':
            #     # TODO: return should be based on the reward function, for now put 1 for a scaled reward
            #     eval_return = self.get_returns(1)

        if not self.wandb_log:
            [print(f"{ep_loss:.5f}") for ep_loss in epoch_losses]  # Debug print
            save_plot({"Loss_per_Epoch": epoch_losses})

    def get_returns(self, ret, k: int = 10, temperature: float = 1.0) -> dict[str, float]:
        self.model.train(False)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        reward_funcs = _as_reward_list(self.reward_func)
        n_goals = len(reward_funcs)
        if raw_model.config.n_goals < n_goals:
            raise ValueError(
                f"Model config n_goals ({raw_model.config.n_goals}) is smaller than reward count ({n_goals})"
            )

        goal_returns: dict[str, float] = {}
        for goal_idx in range(n_goals):
            goal_rewards: list[float] = []
            for _ in range(k):
                terminated = False
                done = False
                init_state = torch.tensor([self.bos_token_id], dtype=torch.int64)
                init_state = init_state.to(self.device).unsqueeze(0).unsqueeze(0)
                rtgs = [[ret]]
                goal = [[goal_idx]]
                # first state is from env, first rtg is target return, and first timestep is 0
                sampled_action = sample(
                    model=self.model,
                    x=init_state,
                    steps=1,
                    temperature=temperature,
                    sample=True,
                    actions=None,
                    rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0),
                    goal=torch.tensor(goal, dtype=torch.int64).to(self.device).unsqueeze(0),
                    # timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device)
                )

                all_states = init_state
                actions = []
                state, reward_sum = [self.bos_token_id], 0
                while True:
                    action = sampled_action.cpu().numpy()[0, -1]
                    actions += [action]
                    state.append(action)
                    sequence = self.train_dataset.dataset.tokenizer.decode(state, skip_special_tokens=True)[0]
                    if self.train_dataset.dataset.string_type == "SELFIES":
                        sequence = sf.decoder(sequence)
                    reward = reward_funcs[goal_idx](sequence)
                    done = action == self.eos_token_id  # mol is complete when [EOS] token is generated
                    reward_sum = reward

                    # if molecule length exceeds block_size and [EOS] token wasn't generated terminate generation
                    if len(state) >= raw_model.config.max_seq_len and not done:
                        terminated = True

                    if done or terminated:
                        goal_rewards.append(reward_sum)
                        break

                    tensor_state = torch.tensor(state, device=self.device).unsqueeze(0).unsqueeze(0)
                    pad_size = tensor_state.shape[-1] - all_states.shape[-1]
                    all_states = torch.nn.functional.pad(all_states, (0, pad_size), value=self.pad_token_id)
                    all_states = torch.cat([all_states, tensor_state], dim=1)

                    rtgs[0].append(rtgs[0][-1] - reward)
                    goal[0].append(goal_idx)
                    # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                    sampled_action = sample(
                        model=self.model,
                        x=all_states,
                        steps=1,
                        temperature=temperature,
                        sample=True,
                        actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(0),
                        rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0),
                        attention=torch.tensor(np.tril(np.ones(all_states.shape[1:])), dtype=torch.long).to(self.device).unsqueeze(0),
                        goal=torch.tensor(goal, dtype=torch.int64).to(self.device).unsqueeze(0),
                        # timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
                    )
            goal_eval_return = float(sum(goal_rewards) / max(1, len(goal_rewards)))
            goal_returns[f"reward/goal_{goal_idx}"] = goal_eval_return
            print(f"goal {goal_idx} target return: {ret}, eval return: {goal_eval_return:.4f}")

        goal_returns["reward/overall"] = float(sum(goal_returns.values()) / max(1, len(goal_returns)))
        print(f"overall target return: {ret}, eval return: {goal_returns['reward/overall']:.4f}")
        self.model.train(True)
        return goal_returns


def run_dt_training(
        model,
        train_dataloader,
        optimizer,
        ctx,
        scaler,
        reward_func,
        save_path,
        train_config,
        test_dataloader=None,
        device: str = "cuda",
        wandb_log=False,
):
    trainer = Trainer(model, train_dataloader, test_dataloader, reward_func, train_config, save_path, device, wandb_log)
    trainer.train(optimizer)
