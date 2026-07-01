"""Functional tests for AgentExecutor with mock LLM adapter."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from miaowa.agent import AgentExecutor, AgentResponse, ContextBuilder
from miaowa.core.config import ConfigManager
from miaowa.core.types import ToolParameter
from miaowa.llm.base import BaseLLMAdapter
from miaowa.llm.tokenizer import TokenCounter
from miaowa.llm.types import ChatResponse, StreamChunk, ToolCall, ModelInfo
from miaowa.prompts.manager import PromptManager
from miaowa.tools.base import BaseTool
from miaowa.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Mock LLM Adapter
# ---------------------------------------------------------------------------

class MockLLMAdapter(BaseLLMAdapter):
    """Simulates LLM behavior with pre-programmed responses."""

    def __init__(self, responses=None, model="mock-model"):
        super().__init__(model=model)
        self.responses = responses or []
        self.call_count = 0

    async def _chat_impl(self, messages, tools):
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp

    async def _stream_impl(self, messages, tools):
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        # Yield content character by character
        if resp.content:
            for ch in resp.content:
                yield StreamChunk(delta_content=ch)

        # Yield tool call deltas
        if resp.tool_calls:
            for i, tc in enumerate(resp.tool_calls):
                arg_str = json.dumps(tc.arguments, ensure_ascii=False)
                yield StreamChunk(tool_call_delta={
                    "index": i, "id": tc.id,
                    "function": {"name": tc.name},
                })
                yield StreamChunk(tool_call_delta={
                    "index": i,
                    "function": {"arguments": arg_str},
                })

        # Final chunk with finish_reason
        fr = "tool_calls" if resp.tool_calls else "stop"
        yield StreamChunk(finish_reason=fr)

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def get_model_info(self):
        return ModelInfo(
            provider="mock", model=self._model or "mock",
            max_tokens=64000, supports_streaming=True, supports_tools=True,
        )


# ---------------------------------------------------------------------------
# Mock Tools
# ---------------------------------------------------------------------------

class EchoTool(BaseTool):
    name = "echo"
    description = "Return the input message"
    parameters = [
        ToolParameter(name="message", type="string",
                       description="Message to echo", required=True),
    ]

    async def execute(self, **kwargs):
        return {"echo": kwargs.get("message", "")}


class AddTool(BaseTool):
    name = "add"
    description = "Sum two numbers"
    parameters = [
        ToolParameter(name="a", type="integer",
                       description="First number", required=True),
        ToolParameter(name="b", type="integer",
                       description="Second number", required=True),
    ]

    async def execute(self, **kwargs):
        return {"result": kwargs["a"] + kwargs["b"]}


class FailingTool(BaseTool):
    name = "failing"
    description = "Always fails"
    parameters = []

    async def execute(self, **kwargs):
        raise RuntimeError("Intentional failure for testing")


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

SAMPLE_PROJECT = (Path(__file__).parent.parent.parent / "fixtures" / "sample_python_project").resolve()

def make_executor(llm_responses, tools=None):
    """Create an AgentExecutor with mock LLM and optional tools."""
    config = ConfigManager.load_default()
    token_counter = TokenCounter("deepseek-chat")
    ctx_builder = ContextBuilder(config, token_counter, PromptManager)

    tool_registry = ToolRegistry()
    if tools:
        for tool in tools:
            tool_registry.register(tool)

    mock_llm = MockLLMAdapter(responses=llm_responses)
    return AgentExecutor(mock_llm, tool_registry, ctx_builder, config=config)


async def collect_run(executor, user_input):
    """Collect all streamed chunks and return full text + executor."""
    chunks = []
    async for chunk in executor.run(user_input, SAMPLE_PROJECT):
        chunks.append(chunk)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_simple_reply():
    """No tool calls — single round, direct reply."""
    print("--- Test 1: Simple reply (no tool calls) ---")
    executor = make_executor(
        llm_responses=[
            ChatResponse(content="Hello! This is a Python project.", finish_reason="stop"),
        ],
    )
    text = await collect_run(executor, "What is this project?")
    resp = executor.last_response

    assert text == "Hello! This is a Python project."
    assert resp.tool_calls_made == 0
    assert resp.iterations == 1
    assert resp.tokens_used["total"] >= 0
    print(f"  Content: {text}")
    print(f"  Tool calls: {resp.tool_calls_made}, Iterations: {resp.iterations}")
    print("  [PASS]")


async def test_single_tool_call():
    """One tool call then final reply."""
    print("\n--- Test 2: Single tool call -> reply ---")
    executor = make_executor(
        llm_responses=[
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="echo",
                                     arguments={"message": "hello"})],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="Tool returned: hello. Done!", finish_reason="stop"),
        ],
        tools=[EchoTool()],
    )
    text = await collect_run(executor, "echo hello")
    resp = executor.last_response

    assert resp.tool_calls_made == 1
    assert resp.iterations == 2
    assert "Done" in text
    print(f"  Content: {text}")
    print(f"  Tool calls: {resp.tool_calls_made}, Iterations: {resp.iterations}")
    print("  [PASS]")


async def test_multiple_tool_calls():
    """Multiple tool calls in one round."""
    print("\n--- Test 3: Multiple tool calls in one round ---")
    executor = make_executor(
        llm_responses=[
            ChatResponse(
                content="Let me do both...",
                tool_calls=[
                    ToolCall(id="c1", name="echo",
                             arguments={"message": "hi"}),
                    ToolCall(id="c2", name="add",
                             arguments={"a": 3, "b": 5}),
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="Both done: echo=hi, add=8", finish_reason="stop"),
        ],
        tools=[EchoTool(), AddTool()],
    )
    text = await collect_run(executor, "do both")
    resp = executor.last_response

    assert resp.tool_calls_made == 2
    assert resp.iterations == 2
    print(f"  Content: {text}")
    print(f"  Tool calls: {resp.tool_calls_made}, Iterations: {resp.iterations}")
    print("  [PASS]")


async def test_tool_not_found():
    """Non-existent tool — error returned to LLM, graceful recovery."""
    print("\n--- Test 4: Non-existent tool -> graceful error ---")
    executor = make_executor(
        llm_responses=[
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="bad", name="nonexistent_tool", arguments={})],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="Sorry, that tool is unavailable.", finish_reason="stop"),
        ],
        tools=[EchoTool()],
    )
    text = await collect_run(executor, "use bad tool")
    resp = executor.last_response

    assert resp.tool_calls_made == 1  # still counted as attempted
    assert "Sorry" in text
    print(f"  Content: {text}")
    print(f"  Tool calls: {resp.tool_calls_made}, Iterations: {resp.iterations}")
    print("  [PASS]")


async def test_tool_execution_error():
    """Tool execution raises exception — captured as error message."""
    print("\n--- Test 5: Tool execution error -> captured gracefully ---")
    executor = make_executor(
        llm_responses=[
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="failing", arguments={})],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="The tool failed, let me try another approach.", finish_reason="stop"),
        ],
        tools=[FailingTool()],
    )
    text = await collect_run(executor, "use failing tool")
    resp = executor.last_response

    assert resp.tool_calls_made == 1
    print(f"  Content: {text}")
    print(f"  Tool calls: {resp.tool_calls_made}, Iterations: {resp.iterations}")
    print("  [PASS]")


async def test_status_and_cache():
    """Test get_status() and invalidate_project_cache()."""
    print("\n--- Test 6: get_status() and invalidate_project_cache() ---")
    executor = make_executor(
        llm_responses=[
            ChatResponse(content="OK", finish_reason="stop"),
        ],
        tools=[EchoTool(), AddTool()],
    )
    await collect_run(executor, "status check")

    status = executor.get_status()
    assert status["tool_count"] == 2
    assert status["max_iterations"] == 10
    assert status["last_response"] is not None
    assert "cache_status" in status
    print(f"  tool_count: {status['tool_count']}")
    print(f"  last_response preview: {status['last_response']['content_preview']}")
    print(f"  cache_status cached: {status['cache_status']['cached']}")

    executor.invalidate_project_cache()
    status2 = executor.get_status()
    print(f"  after invalidate: cached={status2['cache_status']['cached']}")
    print("  [PASS]")


async def test_cost_estimation():
    """Verify cost calculation is reasonable."""
    print("\n--- Test 7: Cost estimation ---")
    executor = make_executor(
        llm_responses=[
            ChatResponse(content="A" * 500, finish_reason="stop"),
        ],
    )
    await collect_run(executor, "cost test")
    resp = executor.last_response

    assert resp.cost >= 0.0
    # With ~1000 prompt + 500 completion tokens, cost should be ~0.002
    print(f"  Cost: {resp.cost}")
    print(f"  Tokens: {resp.tokens_used}")
    assert resp.cost < 0.01  # should be very cheap
    print("  [PASS]")


async def main():
    await test_simple_reply()
    await test_single_tool_call()
    await test_multiple_tool_calls()
    await test_tool_not_found()
    await test_tool_execution_error()
    await test_status_and_cache()
    await test_cost_estimation()
    print("\n=== All functional tests passed ===")


if __name__ == "__main__":
    asyncio.run(main())
