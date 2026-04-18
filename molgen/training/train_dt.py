import math
import os
import re

import numpy as np
import torch
import wandb
from rdkit import Chem
from tqdm import tqdm

from molgen.models.dt_gpt import sample
from molgen.utils.plot_utils import save_plot


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
        wandb_log: bool = False,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.reward_func = reward_func if isinstance(reward_func, list) else [reward_func]
        self.save_path = save_path
        self.config = config
        self.device = device
        self.wandb_log = wandb_log
        self.optimizer = None

        self.bos_token_id = train_dataset.dataset.tokenizer.bos_token_id
        self.eos_token_id = train_dataset.dataset.tokenizer.eos_token_id
        self.pad_token_id = train_dataset.dataset.tokenizer.pad_token_id
        self.ignore_token_id = train_dataset.collate_fn.ignore_index

    def _raw_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _to_device(self, batch):
        if "cuda" in str(self.device):
            return {k: v.pin_memory().to(self.device, non_blocking=True) for k, v in batch.items()}
        return {k: v.to(self.device) for k, v in batch.items()}

    def _calculate_lr(self):
        if not self.config.get("decay_lr", True):
            return self.config["learning_rate"]

        warmup_tokens = self.config.get("warmup_steps", 0)
        if self.tokens < warmup_tokens:
            lr_mult = float(self.tokens) / float(max(1, warmup_tokens))
        else:
            progress = float(self.tokens - warmup_tokens) / float(
                max(1, self.config.get("lr_decay_steps", warmup_tokens * 300) - warmup_tokens)
            )
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return self.config["learning_rate"] * lr_mult

    def save_checkpoint(self, epoch, ckpt_name: str = None):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        if ckpt_name is None:
            ckpt_name = f"epoch_{epoch}.pth" if self.test_dataset is None else "best.pth"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._raw_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "token_counter": self.tokens,
        }
        torch.save(checkpoint, os.path.join(self.save_path, ckpt_name))

    def load_checkpoint(self, ckpt_name: str = None):
        checkpoint_dir = self.save_path
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            return 0, 0

        if ckpt_name is None:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")]
            if len(checkpoint_files) == 0:
                return 0, 0

            epoch_numbers = [
                int(re.search(r"epoch_(\d+)", file).group(1))
                for file in checkpoint_files
                if re.search(r"epoch_(\d+)", file)
            ]
            ckpt_name = f"epoch_{max(epoch_numbers)}.pth"

        checkpoint = torch.load(os.path.join(checkpoint_dir, ckpt_name))
        self._raw_model().load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"] + 1, checkpoint["token_counter"]

    def _decode_generated(self, token_ids):
        tokenizer = self.train_dataset.dataset.tokenizer
        seq = tokenizer.decode(token_ids, skip_special_tokens=True)[0]
        if self.train_dataset.dataset.string_type == "SELFIES":
            import selfies as sf
            seq = sf.decoder(seq)
        return seq

    def _generate_conditioned(self, rtg_targets, goal_idx):
        model = self.model
        model.eval()

        init_state = torch.tensor([self.bos_token_id], dtype=torch.int64, device=self.device).unsqueeze(0).unsqueeze(0)
        rtgs = [[float(x)] for x in rtg_targets]
        goals = list(goal_idx)
        state = [self.bos_token_id]
        all_states = init_state
        actions = []

        max_len = self._raw_model().config.max_seq_len

        for _ in range(max_len):
            rtgs_tensor = torch.tensor(rtgs, dtype=torch.float32, device=self.device).unsqueeze(0)
            goal_tensor = torch.tensor(goals, dtype=torch.int64, device=self.device).unsqueeze(0)

            sampled_action = sample(
                model=model,
                x=all_states,
                steps=1,
                temperature=1.0,
                sample=True,
                actions=torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(0) if len(actions) > 0 else None,
                rtgs=rtgs_tensor,
                attention=torch.tensor(np.tril(np.ones(all_states.shape[1:])), dtype=torch.long, device=self.device).unsqueeze(0),
                goal_idx=goal_tensor,
            )

            action = sampled_action.cpu().numpy()[0, -1]
            actions.append(action)
            state.append(action)

            if action == self.eos_token_id:
                break

            tensor_state = torch.tensor(state, device=self.device).unsqueeze(0).unsqueeze(0)
            pad_size = tensor_state.shape[-1] - all_states.shape[-1]
            all_states = torch.nn.functional.pad(all_states, (0, pad_size), value=self.pad_token_id)
            all_states = torch.cat([all_states, tensor_state], dim=1)

            for i in range(len(rtgs)):
                rtgs[i].append(rtgs[i][-1])

        model.train(True)
        return self._decode_generated(state)

    def _run_generation_validation(self):
        n_samples = self.config.get("validation_samples", 64)
        tolerance = self.config.get("conditioning_tolerance", 0.1)

        single_target = self.config.get("validation_single_goal_target", 0.9)
        multi_targets = self.config.get("validation_multi_goal_targets", [0.8, 0.5])

        generated = []
        abs_errors = []

        for i in range(n_samples):
            if len(self.reward_func) > 1 and i % 2 == 1:
                goals = list(range(min(len(self.reward_func), len(multi_targets))))
                targets = [multi_targets[g] for g in goals]
            else:
                goals = [0]
                targets = [single_target]

            smiles = self._generate_conditioned(targets, goals)
            generated.append(smiles)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            for goal, target in zip(goals, targets):
                try:
                    pred = float(self.reward_func[goal](smiles))
                except Exception:
                    continue
                abs_errors.append(abs(pred - float(target)))

        valid_count = sum(1 for s in generated if Chem.MolFromSmiles(s) is not None)
        validity = valid_count / max(1, len(generated))
        cond_abs_error = float(np.mean(abs_errors)) if len(abs_errors) > 0 else float("inf")
        conditioning_accuracy = float(np.mean(np.array(abs_errors) <= tolerance)) if len(abs_errors) > 0 else 0.0

        return {
            "valid_smiles_rate": validity,
            "conditioning_abs_error": cond_abs_error,
            "conditioning_accuracy": conditioning_accuracy,
        }

    def _run_epoch(self, split: str, epoch_num: int):
        is_train = split == "train"
        self.model.train(is_train)
        loader = self.train_dataset if is_train else self.test_dataset
        if loader is None:
            return 0.0

        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        total_loss = 0.0
        steps = 0

        if is_train:
            self.optimizer.zero_grad()

        iterator = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, batch in iterator:
            batch = self._to_device(batch)
            x = batch["input_ids"]
            y = batch["labels"]
            r = batch["rtgs"]
            a = batch.get("attention_mask", None)
            g = batch.get("goal_idx", None)

            if is_train and self.config.get("decay_lr", True):
                self.tokens += (y != self.ignore_token_id).sum().item()

            with torch.set_grad_enabled(is_train):
                _, loss = self.model(
                    input_ids=x,
                    labels=y,
                    targets=y,
                    rtgs=r,
                    attention_mask=a,
                    goal_idx=g,
                )

                raw_loss = loss
                if is_train:
                    (loss / grad_accum_steps).backward() # gradient accumulation


            total_loss += raw_loss.detach().item()
            steps += 1

            if is_train:
                do_step = ((it + 1) % grad_accum_steps == 0) or ((it + 1) == len(loader))
                if do_step:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("grad_clip", 1.0))
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    lr = self._calculate_lr()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    lr = self.optimizer.param_groups[0]["lr"]

                iterator.set_description(f"epoch {epoch_num + 1} | loss {raw_loss.item():.4f} | lr {lr:e}")

        return total_loss / max(1, steps)

    def train(self, optimizer):
        self.optimizer = optimizer
        if self.config.get("load_checkpoint", False):
            epoch_n, token_n = self.load_checkpoint(ckpt_name="latest.pth")
        else:
            epoch_n, token_n = 0, 0

        self.tokens = token_n
        epochs = self.config["max_steps"] // len(self.train_dataset)
        best_loss = float("inf")
        epoch_losses = []

        for epoch in range(epoch_n, epochs):
            train_loss = self._run_epoch("train", epoch)
            epoch_losses.append(train_loss)
            test_loss = self._run_epoch("test", epoch) if self.test_dataset is not None else train_loss

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }

            if (epoch + 1) % max(1, self.config.get("eval_every", 10)) == 0:
                metrics.update(self._run_generation_validation())

            if self.wandb_log:
                wandb.log(metrics)
            else:
                print(metrics)

            good_model = (epoch > 2 and self.test_dataset is None) or (test_loss < best_loss)
            if self.save_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint(epoch)

        if not self.wandb_log:
            save_plot({"Loss_per_Epoch": epoch_losses})


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
    del ctx, scaler
    trainer = Trainer(model, train_dataloader, test_dataloader, reward_func, train_config, save_path, device, wandb_log)
    trainer.train(optimizer)
