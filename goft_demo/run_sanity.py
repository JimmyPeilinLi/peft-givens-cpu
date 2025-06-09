"""
sanity check：梯度方向、merge/unmerge 等数值一致性验证
"""
import torch, math
from peft import get_peft_model
from givens import GivensConfig

lin = torch.nn.Linear(32, 32, bias=False)
cfg = GivensConfig(strict_oft=True, target_modules=["linear"])
model = get_peft_model(lin, cfg)

x = torch.randn(4, 32, dtype=torch.double, requires_grad=True)
model.double()

def func(inp):
    return model(inp).sum()

# gradcheck
torch.autograd.gradcheck(func, x, eps=1e-6, atol=1e-4)
print("✅ gradcheck passed")

# merge / unmerge
with torch.no_grad():
    y1 = model(x).clone()
merged = model.merge_and_unload()
y2 = merged(x)
assert torch.allclose(y1, y2, atol=1e-5), "merge 后输出不一致"
print("✅ merge equivalence passed")
