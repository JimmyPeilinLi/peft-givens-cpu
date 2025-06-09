import torch, pytest

def test_givens_grad(tiny_model):
    x = torch.randn(16, 64)
    out = tiny_model(x).sum()
    out.backward()
    # 所有 givens_ 参数都有梯度
    for n,p in tiny_model.named_parameters():
        if "givens_" in n:
            assert p.grad is not None, f"{n} grad is None"
