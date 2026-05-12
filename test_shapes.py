import torch
from molgen.models.dt_gpt import DtGPT, DTGPTConfig

device = "cpu"

def test_n_goals(n):
    print(f"\n--- Testing N={n} goals ---")
    config = DTGPTConfig(
        vocab_size=100, 
        n_goals=n, 
        block_size=10, 
        n_embd=128, 
        n_head=4, 
        n_layer=2
    )
    model = DtGPT(config).to(device)
    
    # Batch size 1, N goals, Time steps 5
    rtgs = torch.randn(1, n, 5).to(device)
    goal = torch.randint(0, n, (1, n, 5)).to(device)
    input_ids = torch.randint(0, 100, (1, 5)).to(device)
    
    try:
        logits, loss = model(input_ids=input_ids, rtgs=rtgs, goal=goal, labels=input_ids)
        print(f"Success! Logits shape: {logits.shape}")
        # בדיקה שה-Position Embedding הוקצה נכון
        expected_pos = 5 * (n + 2)
        print(f"Max token positions reserved: {model.max_token_positions}")
    except Exception as e:
        print(f"Failed for N={n}: {e}")

if __name__ == "__main__":
    test_n_goals(1)
    test_n_goals(3)
    test_n_goals(5)
