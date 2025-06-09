import torch

def test_merge_equivalence(tiny_model):
    x = torch.randn(8, 64)
    with torch.no_grad():
        y_base = tiny_model(x)
    merged = tiny_model.merge_and_unload()
    y_merge = merged(x)
    assert torch.allclose(y_base, y_merge, atol=1e-5)
