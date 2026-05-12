#!/usr/bin/env python3
import copy
import datetime as dt
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT_DIR / "data" / "configs" / "dt_config_char_normalized_goal_conditioned.toml"
SWEEP_CONFIG_DIR = ROOT_DIR / "data" / "configs" / "sweeps_tmp"
WANDB_PROJECT = "MolGen-MultiGoal-GradAccum-Sweeps"


@dataclass(frozen=True)
class Experiment:
    name: str
    n_goals: int
    gradient_accumulation_steps: int


EXPERIMENTS: list[Experiment] = [
    Experiment(name="exp01_goals2_accum8", n_goals=2, gradient_accumulation_steps=8),
    Experiment(name="exp02_goals2_accum32", n_goals=2, gradient_accumulation_steps=32),
    Experiment(name="exp03_goals3_accum16", n_goals=3, gradient_accumulation_steps=16),
    Experiment(name="exp04_goals3_accum32", n_goals=3, gradient_accumulation_steps=32),
    Experiment(name="exp05_goals4_accum16", n_goals=4, gradient_accumulation_steps=16),
    Experiment(name="exp06_goals4_accum64", n_goals=4, gradient_accumulation_steps=64),
    Experiment(name="exp07_goals5_accum32", n_goals=5, gradient_accumulation_steps=32),
    Experiment(name="exp08_goals6_accum64", n_goals=6, gradient_accumulation_steps=64),
]

REWARD_TEMPLATES: list[dict[str, Any]] = [
    {"type": "QED"},
    {"type": "PlogP", "scale": {"name": "mult", "factor": 0.1}},
    {"type": "pIC50", "data_path": str(ROOT_DIR / "data" / "datasets" / "properties.csv")},
    {"type": "QED", "scale": {"name": "mult", "factor": 0.75}},
    {"type": "PlogP", "scale": {"name": "mult", "factor": 0.05}},
    {"type": "QED", "scale": {"name": "mult", "factor": 1.25}},
]


def to_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(to_toml_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def write_table(lines: list[str], section: str, table: dict[str, Any]) -> None:
    lines.append(f"[{section}]")
    for key, value in table.items():
        if isinstance(value, dict):
            continue
        lines.append(f"{key} = {to_toml_value(value)}")
    lines.append("")


def dump_config(config: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    write_table(lines, "train_config", config["train_config"])
    write_table(lines, "model_config", config["model_config"])

    reward_cfg = config["reward"]
    lines.append("[reward]")
    for key, value in reward_cfg.items():
        if key == "functions" or isinstance(value, dict):
            continue
        lines.append(f"{key} = {to_toml_value(value)}")

    for reward_fn in reward_cfg["functions"]:
        lines.append("")
        lines.append("[[reward.functions]]")
        for key, value in reward_fn.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    lines.append(f"{key}.{nested_key} = {to_toml_value(nested_value)}")
            else:
                lines.append(f"{key} = {to_toml_value(value)}")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_reward_functions(n_goals: int) -> list[dict[str, Any]]:
    rewards: list[dict[str, Any]] = []
    for i in range(n_goals):
        rewards.append(copy.deepcopy(REWARD_TEMPLATES[i % len(REWARD_TEMPLATES)]))
    return rewards


def build_experiment_config(base_config: dict[str, Any], experiment: Experiment) -> dict[str, Any]:
    reward_functions = build_reward_functions(experiment.n_goals)

    config = copy.deepcopy(base_config)
    config["train_config"]["gradient_accumulation_steps"] = experiment.gradient_accumulation_steps
    config["model_config"]["n_goals"] = experiment.n_goals
    config["reward"]["goal_conditioned"] = True
    config["reward"]["functions"] = reward_functions
    return config


def submit_experiment(config_path: Path, wandb_proj: str, wandb_run_name: str) -> tuple[str | None, str]:
    cmd = ["sbatch", "run_molgen.sbatch", str(config_path), wandb_proj, wandb_run_name]
    proc = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return None, combined.strip()

    match = re.search(r"Submitted batch job (\d+)", combined)
    return (match.group(1) if match else None), combined.strip()


def main() -> int:
    if not BASE_CONFIG_PATH.exists():
        print(f"Base config not found: {BASE_CONFIG_PATH}", file=sys.stderr)
        return 1

    with BASE_CONFIG_PATH.open("rb") as fd:
        base_config = tomllib.load(fd)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SWEEP_CONFIG_DIR / f"sweep_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using base config: {BASE_CONFIG_PATH}")
    print(f"Writing sweep configs to: {run_dir}")
    print(f"W&B Project for all runs: {WANDB_PROJECT}")

    for experiment in EXPERIMENTS:
        exp_config = build_experiment_config(base_config, experiment)
        config_path = run_dir / f"{experiment.name}.toml"
        dump_config(exp_config, config_path)

        wandb_run_name = (
            f"{experiment.name}"
            f"_goals{experiment.n_goals}"
            f"_accum{experiment.gradient_accumulation_steps}"
            f"_{timestamp}"
        )
        job_id, output = submit_experiment(config_path, WANDB_PROJECT, wandb_run_name)

        if job_id is None:
            print(f"[FAILED] {experiment.name} | accum={experiment.gradient_accumulation_steps} | goals={experiment.n_goals}")
            print(f"  Config: {config_path}")
            print(f"  sbatch output: {output}")
        else:
            print(f"[STARTED] {experiment.name} -> Job ID {job_id}")
            print(f"  Config: {config_path}")
            print(f"  W&B Project: {WANDB_PROJECT}")
            print(f"  W&B Run Name: {wandb_run_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
