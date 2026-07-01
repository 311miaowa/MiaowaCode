"""Tests for Planner — intent classification, complexity estimation, think()."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_SAMPLE_PROJECT = (Path(__file__).parent.parent.parent / "fixtures" / "sample_python_project").resolve()

from miaowa.agent.planner import Planner
from miaowa.core.config import ConfigManager
from miaowa.core.types import ThoughtResult, ContextPayload
from miaowa.llm.base import BaseLLMAdapter
from miaowa.llm.tokenizer import TokenCounter
from miaowa.llm.types import (
    Message, ChatResponse, StreamChunk, ToolCall, ModelInfo,
)
from miaowa.agent.context import ContextBuilder
from miaowa.prompts.manager import PromptManager


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLM(BaseLLMAdapter):
    def __init__(self, responses):
        super().__init__(model="mock")
        self.responses = responses
        self.calls = 0

    async def _chat_impl(self, messages, tools):
        resp = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return resp

    async def _stream_impl(self, messages, tools):
        resp = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        if resp.content:
            for ch in resp.content:
                yield StreamChunk(delta_content=ch)
        yield StreamChunk(finish_reason="stop")

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def get_model_info(self):
        return ModelInfo(
            provider="mock", model="mock", max_tokens=64000,
            supports_streaming=True, supports_tools=True,
        )


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------

def test_classify_intent():
    """Verify all intent categories are correctly classified."""
    planner = Planner(MockLLM([]))

    cases = [
        # Quick commands
        ("你好", "quick_command"),
        ("hello", "quick_command"),
        ("帮助", "quick_command"),
        ("help me please", "quick_command"),
        ("你能做什么", "quick_command"),
        ("version", "quick_command"),
        ("谢谢你的帮助", "quick_command"),

        # Project analysis
        ("帮我分析这个项目", "project_analysis"),
        ("这个项目用了什么技术栈", "project_analysis"),
        ("项目结构是怎样的", "project_analysis"),
        ("what is the project structure", "project_analysis"),
        ("dependency analysis", "project_analysis"),
        ("整体架构是什么", "project_analysis"),
        ("构建工具是啥", "project_analysis"),
        ("目录结构", "project_analysis"),
        ("项目规模有多大", "project_analysis"),

        # Code explanation
        ("解释一下这段代码", "code_explanation"),
        ("这个函数是做什么的", "code_explanation"),
        ("explain this code", "code_explanation"),
        ("how does this work", "code_explanation"),
        ("实现原理是什么", "code_explanation"),
        ("代码逻辑", "code_explanation"),
        ("这个类的设计模式", "code_explanation"),

        # File operation
        ("读取 main.py", "file_operation"),
        ("查看文件内容", "file_operation"),
        ("read the file", "file_operation"),
        ("创建一个新的配置文件", "file_operation"),
        ("修改 src/app.py", "file_operation"),
        ("编辑这个文件", "file_operation"),
        ("生成文件", "file_operation"),
        ("write file to disk", "file_operation"),

        # Search
        ("搜索 main 函数", "search"),
        ("查找所有引用", "search"),
        ("find all occurrences", "search"),
        ("grep TODO", "search"),
        ("在哪里定义了 Config", "search"),
        ("who calls this function", "search"),
        ("定位这个错误", "search"),
        ("被谁引用", "search"),

        # General question (fallback)
        ("Python 中如何实现单例模式", "general_question"),
        ("什么是闭包", "general_question"),
        ("推荐一个好的测试框架", "general_question"),
        ("", "general_question"),
    ]

    failed = 0
    for text, expected in cases:
        result = planner.classify_intent(text)
        if result != expected:
            print(f"  FAIL: {text!r} -> expected {expected}, got {result}")
            failed += 1

    passed = len(cases) - failed
    print(f"  classify_intent: {passed}/{len(cases)} passed")
    return failed == 0


# ---------------------------------------------------------------------------
# estimate_complexity
# ---------------------------------------------------------------------------

def test_estimate_complexity():
    """Verify complexity estimation heuristics."""
    planner = Planner(MockLLM([]))

    cases = [
        # Simple
        ("你好", "simple"),
        ("hello", "simple"),
        ("这是什么项目", "simple"),
        ("显示 README", "simple"),

        # Medium
        ("帮我修改配置文件添加一个新的日志级别", "medium"),
        ("fix the bug in auth module", "medium"),
        ("添加单元测试覆盖新的函数", "medium"),
        ("优化这个查询的性能", "medium"),

        # Complex
        ("重构整个项目的错误处理机制，统一使用自定义异常类", "complex"),
        ("分析整个项目的架构并给出优化建议和实施方案", "complex"),
        (
            "1. 修改配置文件 2. 更新文档 3. 添加单元测试 4. 执行部署",
            "complex",
        ),
        (
            "这个项目用了什么框架？怎么配置的？为什么选择它？有没有更好的替代方案？",
            "complex",
        ),
        ("migrate the entire codebase from JavaScript to TypeScript", "complex"),
        ("实现一个新的用户认证系统包含登录注册和权限管理", "complex"),
    ]

    failed = 0
    for text, expected in cases:
        result = planner.estimate_complexity(text)
        if result != expected:
            print(f"  FAIL: {text!r:60s} -> expected {expected}, got {result}")
            failed += 1

    passed = len(cases) - failed
    print(f"  estimate_complexity: {passed}/{len(cases)} passed")
    return failed == 0


# ---------------------------------------------------------------------------
# V2 reserved interfaces
# ---------------------------------------------------------------------------

async def test_v2_interfaces():
    """Verify V2 methods raise NotImplementedError."""
    planner = Planner(MockLLM([]))

    try:
        await planner.plan_explicit("", None)
        assert False, "should raise"
    except NotImplementedError:
        print("  [PASS] plan_explicit() raises NotImplementedError")

    try:
        await planner.revise_plan({}, {})
        assert False, "should raise"
    except NotImplementedError:
        print("  [PASS] revise_plan() raises NotImplementedError")

    try:
        planner.get_plan_progress()
        assert False, "should raise"
    except NotImplementedError:
        print("  [PASS] get_plan_progress() raises NotImplementedError")

    return True


# ---------------------------------------------------------------------------
# think()
# ---------------------------------------------------------------------------

async def test_think_direct_reply():
    """LLM returns direct reply → ThoughtResult with thought, no tool_calls."""
    print("  --- think(): LLM direct reply ---")
    mock_llm = MockLLM([
        ChatResponse(
            content="这是一个 Python 项目，使用了 FastAPI 框架。",
            finish_reason="stop",
        ),
    ])
    planner = Planner(mock_llm)
    config = ConfigManager.load_default()
    token_counter = TokenCounter("deepseek-chat")
    ctx_builder = ContextBuilder(config, token_counter, PromptManager)

    context = await ctx_builder.build(
        "这是什么项目", [],
        _SAMPLE_PROJECT, [],
    )

    result = await planner.think("这是什么项目", context)

    assert isinstance(result, ThoughtResult)
    assert result.tool_calls is None
    assert "Python" in (result.thought or "")
    print(f"    thought: {(result.thought or '')[:60]}...")
    print("    [PASS]")
    return True


async def test_think_tool_calls():
    """LLM requests tool calls → ThoughtResult with tool_calls."""
    print("  --- think(): LLM requests tools ---")
    mock_llm = MockLLM([
        ChatResponse(
            content="我需要先读取文件。",
            tool_calls=[
                ToolCall(id="c1", name="read_file",
                         arguments={"file_path": "main.py"}),
            ],
            finish_reason="tool_calls",
        ),
    ])
    planner = Planner(mock_llm)
    config = ConfigManager.load_default()
    token_counter = TokenCounter("deepseek-chat")
    ctx_builder = ContextBuilder(config, token_counter, PromptManager)

    context = await ctx_builder.build(
        "show main.py", [],
        _SAMPLE_PROJECT, [],
    )

    result = await planner.think("show main.py", context)

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"file_path": "main.py"}
    print(f"    tool_calls: {len(result.tool_calls)} call(s)")
    print(f"    first tool: {result.tool_calls[0].name}")
    print("    [PASS]")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    results = []

    print("=== classify_intent ===")
    results.append(test_classify_intent())

    print("\n=== estimate_complexity ===")
    results.append(test_estimate_complexity())

    print("\n=== V2 Reserved Interfaces ===")
    results.append(await test_v2_interfaces())

    print("\n=== think() ===")
    results.append(await test_think_direct_reply())
    results.append(await test_think_tool_calls())

    all_pass = all(results)
    print(f"\n{'=== ALL TESTS PASSED ===' if all_pass else '=== SOME TESTS FAILED ==='}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
