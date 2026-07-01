"""Sample App 测试。"""

from sample_app.main import add, fibonacci


def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0


def test_fibonacci():
    assert fibonacci(0) == []
    assert fibonacci(1) == [0]
    assert fibonacci(5) == [0, 1, 1, 2, 3]
