"""Context Builder — 构建发送给 LLM 的完整上下文。

PRD §5.2.2: Context Builder 负责组装 system prompt、项目分析上下文、
对话历史和当前用户消息，并进行 Token 预算检查与截断。

核心流程::

    1. 构建 system prompt（含当前目录、可用工具描述）
    2. 构建 project context（首次调用时分析并缓存，后续复用）
    3. 组装 messages → [system, project_context, ...history, user_message]
    4. Token 预算检查 — 超限时触发智能截断（保留 system + 最后一条 user）

截断策略委托给 ``TokenCounter.truncate_messages()``，其保证：
    - system 消息（首条）始终保留
    - 最后一条 user 消息始终保留
    - tool-call ↔ tool-result 成对原子保留/删除
    - 中间消息从旧到新保留至预算耗尽
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from miaowa.core.config import Config
from miaowa.core.logger import get_logger
from miaowa.core.types import ContextPayload, ProjectCache
from miaowa.llm.tokenizer import TokenCounter, _MSG_OVERHEAD_TOKENS
from miaowa.llm.types import Message
from miaowa.prompts.manager import PromptManager
from miaowa.tools.analyzer import ProjectAnalyzer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 模块常量
# ---------------------------------------------------------------------------

# 项目缓存 TTL（秒）。0 表示禁用自动过期，仅依赖手动 invalidate_cache()。
# 对于频繁变更的项目建议设为 300（5 分钟）；静态项目可设为 0。
CACHE_TTL_SECONDS: float = 300.0


class ContextBuilder:
    """上下文构建器 — 组装 LLM 输入上下文并管理 Token 预算。

    将系统提示词、项目分析结果、对话历史和当前用户消息组装为
    完整的消息列表，在超出 ``MAX_CONTEXT_TOKENS`` 时触发截断。

    Token 预算模型（以 DeepSeek V4 1M 上下文窗口为例）::

        ┌─────────────────────────────────────────────┬──────────────────┐
        │          ContextBuilder 负责 (≤128K)          │    模型输出       │
        │  system │ project │ history │ user_message   │  (预留 ~872K)   │
        └─────────────────────────────────────────────┴──────────────────┘

    Attributes:
        MAX_CONTEXT_TOKENS: 输入上下文 Token 预算上限（类常量，默认 128,000）。
    """

    MAX_CONTEXT_TOKENS: int = 128_000
    """输入上下文 Token 预算上限。

    以 DeepSeek V4 1M 上下文窗口为基准，默认使用 128K（~13% 窗口），
    为 384K 最大输出和 project context 增长预留充足空间。
    可通过子类化或直接修改类属性调整。
    """

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Config,
        token_counter: TokenCounter,
        prompt_manager: type[PromptManager] = PromptManager,
    ) -> None:
        """初始化上下文构建器。

        Args:
            config: Miaowa 应用配置对象。
            token_counter: Token 计数器实例，提供 count_tokens /
                count_messages / truncate_messages 方法。
            prompt_manager: ``PromptManager`` 类（其方法均为类方法，
                无需实例化直接传入类对象即可）。
        """
        self._config = config
        self._token_counter = token_counter
        self._prompt_manager = prompt_manager

        # 项目分析缓存
        self._project_cache: ProjectCache | None = None
        self._cache_built_at: float | None = None
        self._cached_project_root: Path | None = None

        logger.info(
            "ContextBuilder 初始化完成: "
            f"MAX_CONTEXT_TOKENS={self.MAX_CONTEXT_TOKENS}"
        )

    # ------------------------------------------------------------------
    # build — 核心公共方法
    # ------------------------------------------------------------------

    async def build(
        self,
        user_message: str,
        history: list[Message],
        project_root: Path,
        tool_definitions: list[dict[str, Any]],
    ) -> ContextPayload:
        """构建发送给 LLM 的完整上下文。

        这是 ContextBuilder 的主入口方法，按以下四步执行：

        1. **构建 system prompt** — 含当前目录、可用工具描述
        2. **构建 project context** — 首次调用时运行项目分析并缓存结果
        3. **组装 messages** — ``[system, project_context, ...history, user_message]``
        4. **Token 预算检查** — 超出 ``MAX_CONTEXT_TOKENS`` 时触发截断

        Args:
            user_message: 当前用户输入的文本（纯文本，不含 role 包装）。
            history: 对话历史消息列表。应为按时间升序排列的
                user / assistant / tool 消息，不包含 system 消息。
            project_root: 项目根目录的绝对路径。
            tool_definitions: 可用工具的 OpenAI / DeepSeek Function Calling
                格式定义列表。每项结构::

                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "读取指定文件内容",
                            "parameters": { ... }
                        }
                    }

        Returns:
            ContextPayload: 包含组装后的消息列表和近似 Token 总数。
                调用方可将 ``payload.messages`` 直接传给 LLM 适配器的
                ``chat()`` / ``stream()`` 方法。

        Raises:
            RuntimeError: 即使只保留 system prompt、project context
                和最后一条 user 消息，仍超出 Token 预算时抛出。
                调用方应捕获此异常并考虑缩减 system prompt 或
                增大 ``MAX_CONTEXT_TOKENS``。
        """
        logger.info(
            f"开始构建上下文: "
            f"project_root={project_root}, "
            f"history_len={len(history)}, "
            f"tool_count={len(tool_definitions)}, "
            f"user_msg_len={len(user_message)}"
        )

        # -- 步骤 1: 构建 system prompt ----------------------------------
        system_prompt = self._build_system_prompt(
            current_dir=str(project_root),
            tool_definitions=tool_definitions,
        )
        system_tokens = self._token_counter.count_tokens(system_prompt)
        logger.info(
            f"[步骤 1/4] System prompt: ~{system_tokens} tokens, "
            f"{len(tool_definitions)} 个工具描述已嵌入"
        )

        # -- 步骤 2: 构建 project context（带缓存）------------------------
        await self._build_project_cache(project_root)
        project_context = self._format_project_context()
        project_tokens = self._token_counter.count_tokens(project_context)
        logger.info(
            f"[步骤 2/4] Project context: ~{project_tokens} tokens, "
            f"缓存={'有效' if self._project_cache else '空'}"
        )

        # -- 步骤 3: 组装 messages ---------------------------------------
        messages: list[Message] = []

        # 系统消息（主 system prompt）
        messages.append(Message(role="system", content=system_prompt))

        # 项目上下文（独立 system 消息，方便截断时按粒度处理）
        if project_context:
            messages.append(Message(role="system", content=project_context))

        # 对话历史
        messages.extend(history)

        # 当前用户消息（空消息跳过，避免浪费 tokens）
        if user_message:
            messages.append(Message(role="user", content=user_message))

        total_before = self._token_counter.count_messages(messages)
        logger.info(
            f"[步骤 3/4] 消息组装完成: {len(messages)} 条, "
            f"~{total_before} tokens"
        )

        # -- 步骤 4: Token 预算检查与截断 --------------------------------
        if total_before > self.MAX_CONTEXT_TOKENS:
            logger.warning(
                f"[步骤 4/4] 上下文超出预算: "
                f"{total_before} > {self.MAX_CONTEXT_TOKENS} tokens — 开始截断"
            )
            try:
                original_count = len(messages)
                messages = self._token_counter.truncate_messages(
                    messages, self.MAX_CONTEXT_TOKENS
                )
                total_after = self._token_counter.count_messages(messages)
                removed = original_count - len(messages)
                logger.info(
                    f"[步骤 4/4] 截断完成: "
                    f"{total_before} → ~{total_after} tokens, "
                    f"移除 ~{max(0, removed)} 条中间历史消息, "
                    f"保留 {len(messages)} 条"
                )
            except RuntimeError:
                # 即使只保留必留消息仍超预算 → 记录诊断信息后重新抛出
                must_keep_estimate = (
                    system_tokens
                    + project_tokens
                    + self._token_counter.count_tokens(user_message)
                    + 3 * _MSG_OVERHEAD_TOKENS
                )
                logger.error(
                    f"[步骤 4/4] 上下文严重超预算，无法截断至 "
                    f"{self.MAX_CONTEXT_TOKENS} tokens。"
                    f"最低必需: ~{must_keep_estimate} tokens "
                    f"(system={system_tokens}, project={project_tokens}, "
                    f"user={self._token_counter.count_tokens(user_message)}, "
                    f"overhead={3 * _MSG_OVERHEAD_TOKENS})"
                )
                raise
        else:
            total_after = total_before
            usage_pct = total_after * 100 // self.MAX_CONTEXT_TOKENS
            logger.info(
                f"[步骤 4/4] 上下文在预算内: "
                f"~{total_after}/{self.MAX_CONTEXT_TOKENS} tokens ({usage_pct}%)"
            )

        return ContextPayload(messages=messages, total_tokens=total_after)

    # ------------------------------------------------------------------
    # 项目缓存构建
    # ------------------------------------------------------------------

    async def _build_project_cache(self, project_root: Path) -> None:
        """构建项目分析缓存（仅在首次调用或项目根目录变更时执行）。

        委托 ``ProjectAnalyzer.full_analysis()`` 执行完整项目分析，
        包括技术栈检测、模块分析、依赖提取、架构识别和目录树生成。

        缓存策略：
            - 首次调用（缓存为 ``None``）→ 执行完整分析
            - 同一 ``project_root`` → 复用缓存，跳过分析
            - ``project_root`` 变更 → 自动重建缓存

        若分析过程抛出异常，将缓存设为 ``None``（跳过项目上下文），
        确保后续流程不因分析失败而中断。

        Args:
            project_root: 项目根目录的绝对路径。
        """
        # 缓存命中：同一个项目且在 TTL 内
        if (
            self._project_cache is not None
            and self._cached_project_root == project_root
        ):
            age = self._get_cache_age()
            if CACHE_TTL_SECONDS <= 0 or (age is not None and age < CACHE_TTL_SECONDS):
                logger.info(f"项目缓存命中 (已缓存 {age:.0f}s): {project_root}")
                return
            else:
                logger.info(
                    f"项目缓存已过期 (已缓存 {age:.0f}s > TTL={CACHE_TTL_SECONDS}s)，重建"
                )

        # 项目根目录变更 → 旧缓存失效
        if (
            self._project_cache is not None
            and self._cached_project_root != project_root
        ):
            logger.info(
                f"项目根目录变更: "
                f"{self._cached_project_root} → {project_root}，重建缓存"
            )

        # 执行完整项目分析
        logger.info(f"开始项目分析: {project_root}")
        t_start = time.time()

        try:
            analyzer = ProjectAnalyzer(
                project_root=project_root,
                config=self._config,
            )
            analysis_result = await analyzer.full_analysis()

            elapsed = time.time() - t_start

            # 将分析结果映射到 ProjectCache dataclass
            tech_stack = analysis_result.get("tech_stack", {})
            statistics = analysis_result.get("statistics", {})

            self._project_cache = ProjectCache(
                tech_stack={
                    "languages": tech_stack.get("languages", []),
                    "frameworks": tech_stack.get("frameworks", {}),
                    "build_tools": tech_stack.get("build_tools", []),
                    "package_manager": tech_stack.get("package_manager"),
                },
                structure={
                    "name": analysis_result.get(
                        "project_name", project_root.name
                    ),
                    "architecture": analysis_result.get("structure", "未知"),
                    "tree": analysis_result.get("directory_tree", ""),
                    "module_count": len(analysis_result.get("modules", [])),
                    "file_count": statistics.get("total_files", 0),
                },
                key_files=analysis_result.get("key_files", []),
            )

            self._cache_built_at = time.time()
            self._cached_project_root = project_root

            cache = self._project_cache
            logger.info(
                f"项目分析完成 (耗时 {elapsed:.1f}s): "
                f"语言={cache.tech_stack.get('languages', [])}, "
                f"架构={cache.structure.get('architecture')}, "
                f"文件={cache.structure.get('file_count')}, "
                f"模块={cache.structure.get('module_count')}, "
                f"关键文件={len(cache.key_files)}"
            )

        except Exception:
            elapsed = time.time() - t_start
            logger.exception(
                f"项目分析失败 (耗时 {elapsed:.1f}s)，跳过项目上下文"
            )
            self._project_cache = None
            self._cache_built_at = None
            self._cached_project_root = None

    # ------------------------------------------------------------------
    # 格式化方法
    # ------------------------------------------------------------------

    def _format_project_context(self) -> str:
        """将项目缓存格式化为 LLM 可读的上下文字符串。

        输出 Markdown 格式，包含项目名称、技术栈、架构风格、
        目录结构和关键文件列表。缓存为空时返回空字符串。

        Returns:
            格式化后的项目上下文文本。无缓存或缓存为空时返回 ``""``。
        """
        if self._project_cache is None:
            return ""

        cache = self._project_cache
        lines: list[str] = []
        project_name = cache.structure.get("name", "未知项目")

        lines.append(f"## 当前项目: {project_name}")
        lines.append("")

        # -- 技术栈 -------------------------------------------------------
        languages = cache.tech_stack.get("languages", [])
        if languages:
            lines.append(f"- **主要语言**: {', '.join(languages)}")

        frameworks = cache.tech_stack.get("frameworks", {})
        if frameworks:
            fw_parts = [
                f"{lang}: {', '.join(fws)}"
                for lang, fws in frameworks.items()
            ]
            lines.append(f"- **框架**: {'; '.join(fw_parts)}")

        build_tools = cache.tech_stack.get("build_tools", [])
        if build_tools:
            lines.append(f"- **构建工具**: {', '.join(build_tools)}")

        pm = cache.tech_stack.get("package_manager")
        if pm:
            lines.append(f"- **包管理器**: {pm}")

        # -- 架构与规模 --------------------------------------------------
        architecture = cache.structure.get("architecture", "未知")
        file_count = cache.structure.get("file_count", 0)
        module_count = cache.structure.get("module_count", 0)
        lines.append(f"- **代码架构**: {architecture}")
        lines.append(
            f"- **项目规模**: {file_count} 个文件, {module_count} 个模块"
        )

        # -- 目录结构 ----------------------------------------------------
        tree = cache.structure.get("tree", "")
        if tree:
            lines.append("")
            lines.append("### 目录结构")
            lines.append("")
            lines.append("```")
            lines.append(tree)
            lines.append("```")

        # -- 关键文件 ----------------------------------------------------
        key_files = cache.key_files
        if key_files:
            lines.append("")
            lines.append("### 关键文件")
            lines.append("")
            for f in key_files:
                lines.append(f"- `{f}`")

        result = "\n".join(lines)
        return result if result.strip() else ""

    def _build_system_prompt(
        self,
        current_dir: str,
        tool_definitions: list[dict[str, Any]],
    ) -> str:
        """构建完整的 system prompt（基础模板 + 工具列表）。

        从 ``PromptManager`` 获取基础系统提示词，
        追加可用工具的简要描述列表（供 LLM 理解工具能力边界）。

        Args:
            current_dir: 当前工作目录路径。
            tool_definitions: OpenAI 格式的工具定义列表。

        Returns:
            完整的 system prompt 字符串。
        """
        base = self._prompt_manager.get_system_prompt(
            current_dir,
            provider=self._config.llm.provider,
            model=self._config.llm.model,
        )

        if not tool_definitions:
            return base

        parts: list[str] = [base, "", "## 可用工具", ""]
        for tool in tool_definitions:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            parts.append(f"- **{name}**: {desc}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 缓存管理 API
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """使项目缓存失效。

        下次调用 ``build()`` 时将重新执行完整项目分析。
        适用于以下场景：
            - 项目文件结构发生重大变更（新增/删除模块）
            - 用户通过 ``/cd`` 切换工作目录
            - 手动请求刷新项目上下文
        """
        if self._project_cache is not None:
            age = self._get_cache_age()
            logger.info(
                f"项目缓存已手动失效 "
                f"(原项目: {self._cached_project_root}, "
                f"缓存存活: {age:.0f}s)"
            )
        else:
            logger.info("项目缓存已为空，invalidate_cache 无操作")

        self._project_cache = None
        self._cache_built_at = None
        self._cached_project_root = None

    def get_cache_status(self) -> dict[str, Any]:
        """获取项目缓存的状态信息。

        可供状态栏、调试命令或日志系统查询缓存健康状态。

        Returns:
            dict 包含以下键:
                - ``cached`` (bool): 是否存在有效缓存。
                - ``project_root`` (str | None): 缓存对应的项目根目录路径。
                - ``age_seconds`` (float | None): 缓存存活时间（秒），
                  无缓存时为 None。
                - ``languages`` (list[str]): 检测到的编程语言列表。
                - ``file_count`` (int): 项目文件总数。
                - ``architecture`` (str | None): 代码架构风格名称。
        """
        status: dict[str, Any] = {
            "cached": self._project_cache is not None,
            "project_root": (
                str(self._cached_project_root)
                if self._cached_project_root
                else None
            ),
            "age_seconds": self._get_cache_age(),
        }

        if self._project_cache is not None:
            status.update({
                "languages": self._project_cache.tech_stack.get(
                    "languages", []
                ),
                "file_count": self._project_cache.structure.get(
                    "file_count", 0
                ),
                "architecture": self._project_cache.structure.get(
                    "architecture"
                ),
            })
        else:
            status.update({
                "languages": [],
                "file_count": 0,
                "architecture": None,
            })

        return status

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_cache_age(self) -> float | None:
        """获取项目缓存的存活时间（秒）。

        Returns:
            缓存已存在的时间，单位为秒。无缓存时返回 ``None``。
        """
        if self._cache_built_at is None:
            return None
        return time.time() - self._cache_built_at
