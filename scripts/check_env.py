"""Print the environment required to reproduce cudarepo experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cudarepo.environment import collect_environment


def main() -> None:
    report = collect_environment()
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    report["bundled_nvrtc"] = [
        path.name
        for path in sorted(torch_lib.glob("nvrtc64_*.dll"))
        if "builtins" not in path.name and ".alt." not in path.name
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA不可用：请确认已安装CUDA版PyTorch且驱动可见。")
    if not report["bundled_nvrtc"]:
        raise SystemExit("当前PyTorch wheel中没有找到NVRTC动态库。")


if __name__ == "__main__":
    main()

