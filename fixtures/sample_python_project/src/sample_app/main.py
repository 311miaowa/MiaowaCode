"""Sample App 入口模块。"""

import sys


def main() -> None:
    """示例应用入口。"""
    name = sys.argv[1] if len(sys.argv) > 1 else "World"
    print(f"Hello, {name}!")


def add(a: int, b: int) -> int:
    """简单加法函数。"""
    return a + b


def fibonacci(n: int) -> list[int]:
    """生成斐波那契数列。"""
    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


if __name__ == "__main__":
    main()
