import importlib, pathlib, torch, types
from types import ModuleType

# 1) 直接尝试 import 已安装的 goft_cpu_ext
try:
    _mod = importlib.import_module("goft_cpu_ext")
except ImportError:
    # 2) fallback：加载同目录下的 .so（本地开发模式）
    _so = pathlib.Path(__file__).with_suffix(".so")
    if not _so.exists():
        raise ImportError(
            "GOFT CPU extension not found. "
            "Run: cd tuners/givens/ext_cpu && python setup.py install"
        )
    torch.ops.load_library(str(_so))
    _mod = importlib.import_module("goft_cpu_ext")

# 暴露给外部
rot_autograd = _mod.rot_autograd
rot_stack_autograd = getattr(_mod, "rot_stack_autograd", None)