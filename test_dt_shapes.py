import torch

from molgen.models.dt_gpt import DTGPTConfig, DtGPT


def run_shape_check(n_goals: int, batch: int = 1, time_steps: int = 5, vocab_size: int = 100) -> None:
    config = DTGPTConfig(
        vocab_size=vocab_size,
        n_goals=n_goals,
        block_size=12,
        max_seq_len=12,
        n_embd=128,
        n_head=4,
        n_layer=2,
        model_type="reward_conditioned",
    )
    model = DtGPT(config).to("cpu")
    model.eval()

    # 2D states are intentionally used to verify dynamic input normalization in forward().
    input_ids = torch.randint(0, vocab_size, (batch, time_steps), dtype=torch.int64, device="cpu")
    labels = torch.randint(0, vocab_size, (batch, time_steps), dtype=torch.int64, device="cpu")
    rtgs = torch.randn(batch, n_goals, time_steps, dtype=torch.float32, device="cpu")
    goal = torch.randint(0, n_goals, (batch, n_goals, time_steps), dtype=torch.int64, device="cpu")

    with torch.no_grad():
        logits, _ = model(input_ids=input_ids, labels=labels, targets=labels, rtgs=rtgs, goal=goal)

    expected_shape = (batch, time_steps, vocab_size)
    assert logits.shape == expected_shape, f"Expected logits {expected_shape}, got {tuple(logits.shape)}"

    expected_tokens = time_steps * (n_goals + 2)
    assert expected_tokens <= model.max_token_positions, (
        f"Position capacity too small: tokens={expected_tokens}, max={model.max_token_positions}"
    )
    print(f"Success N={n_goals}: logits={tuple(logits.shape)}, max_pos={model.max_token_positions}")


if __name__ == "__main__":
    for n in (1, 3, 5):
        run_shape_check(n)
