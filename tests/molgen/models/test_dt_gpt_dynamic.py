import pytest
import torch

from molgen.models.dt_gpt import DTGPTConfig, DtGPT


@pytest.mark.parametrize("n_goals", [1, 3, 5])
def test_dt_forward_dynamic_goals_with_2d_inputs(n_goals: int):
    config = DTGPTConfig(
        vocab_size=101,
        block_size=24,
        max_seq_len=24,
        n_embd=64,
        n_head=4,
        n_layer=2,
        n_goals=n_goals,
        model_type="reward_conditioned",
        dropout=0.0,
    )
    model = DtGPT(config).to("cpu")
    model.eval()

    batch, time_steps = 1, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, time_steps), dtype=torch.int64)
    labels = torch.randint(0, config.vocab_size, (batch, time_steps), dtype=torch.int64)
    rtgs = torch.randn(batch, n_goals, time_steps, dtype=torch.float32)
    goal = torch.randint(0, n_goals, (batch, n_goals, time_steps), dtype=torch.int64)

    logits, loss = model(input_ids=input_ids, labels=labels, targets=labels, rtgs=rtgs, goal=goal)

    assert logits.shape == (batch, time_steps, config.vocab_size)
    assert loss is not None


def test_dt_forward_with_3d_state_and_attention_mask():
    n_goals = 3
    config = DTGPTConfig(
        vocab_size=89,
        block_size=24,
        max_seq_len=24,
        n_embd=64,
        n_head=4,
        n_layer=2,
        n_goals=n_goals,
        model_type="reward_conditioned",
        dropout=0.0,
    )
    model = DtGPT(config).to("cpu")
    model.eval()

    batch, time_steps, state_len = 1, 5, 5
    input_ids = torch.randint(0, config.vocab_size, (batch, time_steps, state_len), dtype=torch.int64)
    labels = torch.randint(0, config.vocab_size, (batch, time_steps), dtype=torch.int64)
    rtgs = torch.randn(batch, n_goals, time_steps, dtype=torch.float32)
    goal = torch.randint(0, n_goals, (batch, n_goals, time_steps), dtype=torch.int64)

    # Per-timestep causal mask across state tokens.
    attention_mask = torch.zeros((batch, time_steps, state_len), dtype=torch.int64)
    for t in range(time_steps):
        attention_mask[:, t, : t + 1] = 1

    logits, _ = model(
        input_ids=input_ids,
        labels=labels,
        targets=labels,
        rtgs=rtgs,
        goal=goal,
        attention_mask=attention_mask,
    )
    assert logits.shape == (batch, time_steps, config.vocab_size)


def test_dt_forward_rejects_excess_goals():
    config = DTGPTConfig(
        vocab_size=51,
        block_size=16,
        max_seq_len=16,
        n_embd=32,
        n_head=4,
        n_layer=2,
        n_goals=2,
        model_type="reward_conditioned",
        dropout=0.0,
    )
    model = DtGPT(config).to("cpu")
    batch, time_steps = 1, 4
    input_ids = torch.randint(0, config.vocab_size, (batch, time_steps), dtype=torch.int64)
    labels = torch.randint(0, config.vocab_size, (batch, time_steps), dtype=torch.int64)
    rtgs = torch.randn(batch, 3, time_steps, dtype=torch.float32)
    goal = torch.randint(0, 3, (batch, 3, time_steps), dtype=torch.int64)

    with pytest.raises(ValueError, match="model was initialized with 2"):
        model(input_ids=input_ids, labels=labels, targets=labels, rtgs=rtgs, goal=goal)
