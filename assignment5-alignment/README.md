# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)

```
uv sync --no-install-package flash-attn
uv sync
# 为了避免wsl环境汇总拉取github失败，推荐使用 uv sync --frozen
```

2. Run unit tests:

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

（请勿随意修改此文件，此为原项目自带）

wsl中执行命令前先激活环境：source .venv/bin/activate

win执行uv命令前先激活：

```Shell
.venv-win\Scripts\activate
```

 进入python后直接执行python脚本：`python main.py`
