import torch, pytest
from peft import get_peft_model
from givens import GivensConfig

@pytest.fixture(scope="session")
def tiny_model():
    lin = torch.nn.Linear(64, 64, bias=False)
    cfg = GivensConfig(strict_oft=True, target_modules=["linear"])
    return get_peft_model(lin, cfg)
