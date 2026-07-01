"""任务规划器 — 分析用户意图，制定执行策略。

PRD §5.2.1: Planner 负责意图分类、复杂度评估和 LLM 决策请求。

MVP 策略（隐式规划）::

    PRD §2.5 定义了 Understand → Plan → Execute → Synthesize 四阶段。
    MVP 将 Phase 1 (Understand) 和 Phase 2 (Plan) 合并为一次 LLM 推理调用，
    即「隐式规划」：模型在生成回复文本或工具调用时自然融入了规划逻辑。

    Planner 在 MVP 中扮演**轻量预分析**角色：
        - classify_intent() — 本地关键词分类（零延迟）
        - estimate_complexity() — 本地启发式评估
        - think() — 调用 LLM 获取初始决策（ThoughtResult），
          决策中可能包含 tool_calls（模型已自主规划所需工具）

V2 演进方向（预留接口）::

    - plan_explicit() — 生成结构化的多步执行计划
    - revise_plan() — 根据工具执行结果动态修正计划
    - get_plan_progress() — 查询当前计划执行进度
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from miaowa.core.logger import get_logger
from miaowa.core.types import ContextPayload, ThoughtResult
from miaowa.llm.base import BaseLLMAdapter
from miaowa.llm.types import Message

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 意图分类关键词库
# ---------------------------------------------------------------------------
# 每个类别包含中文关键词和英文模式。匹配时对输入做大小写不敏感处理。
# 设计原则：
#   - 关键词 ≥ 3 字符（"搜索"/"查找"/"重构"/"grep"/"import" 五个高语义明确词除外）
#   - 每个类别的关键词互不重叠，减少歧义
#   - 英文模式使用正则单词边界避免误匹配（如 "find" 不匹配 "finding"）

# 得分制匹配：中文子串命中 +1 分，英文正则命中 +2 分；最高分胜出。

_INTENT_RULES: list[tuple[str, list[str], list[str]]] = [
    # (category, chinese_keywords, english_patterns)
    # 设计原则：所有中文关键词 ≥ 3 字符，降低子串误匹配率。
    (
        "project_analysis",
        [
            "项目结构", "项目分析", "技术栈", "用了什么技术", "用了什么框架",
            "项目概述", "项目简介", "整体架构", "代码架构", "模块结构",
            "依赖关系", "技术选型", "目录结构", "项目规模", "架构风格",
            "架构设计", "是什么项目", "什么语言写的", "构建工具", "包管理器",
            "分析项目", "项目信息", "结构分析", "项目总览",
            "分析这个项目", "分析一下项目", "这个项目", "当前项目",
            "用了什么", "用了哪些", "项目里", "都有什么", "有哪些",
            "重构",  # 语义高度明确（项目级重构），低误匹配风险
            "重构方案", "架构调整", "代码重构", "技术升级",
            "解释项目", "了解项目", "介绍项目",
        ],
        [
            r"\bproject\s+(structure|overview|analysis|layout|info)\b",
            r"\btech\s+stack\b",
            r"\b(architecture|dependency|dependencies)\b",
            r"\bwhat\s+(is|are)\s+(this|the)\s+project\b",
            r"\brefactor(ing)?\b",
        ],
    ),
    (
        "code_explanation",
        [
            "这段代码", "这个函数", "这个类", "这个模块",
            "什么意思", "做什么的", "如何工作", "工作机制",
            "实现原理", "代码逻辑", "运行流程", "调用链", "调用关系",
            "设计模式", "为什么这样写", "代码作用", "功能说明",
            "方法说明", "类说明",
            "解释一下", "解释这个", "帮我解释", "帮忙解释",
            "解释代码", "解释原理", "解释逻辑",
        ],
        [
            r"\bexplain\b",
            r"\bhow\s+does\b",
            r"\bwhat\s+does\b",
            r"\bcode\s+(logic|flow|review)\b",
        ],
    ),
    (
        "file_operation",
        [
            "查看文件", "显示文件", "创建文件", "新建文件",
            "写一个", "写个文件", "写入文件", "输出到", "保存到", "生成文件",
            "编辑文件", "修改文件", "修改代码", "修改配置", "修改一下",
            "改一下", "改成新的", "更新文件", "删除文件",
            "复制文件", "移动文件", "追加内容",
            "读取文件", "打开文件", "重命名文件", "新建目录",
        ],
        [
            r"\b(read|write|create|edit|modify|update|delete|rename|copy|move)\s+(the\s+)?file\b",
            r"\bshow\s+(me\s+)?(the\s+)?(file|content)\b",
        ],
    ),
    (
        "search",
        [
            "搜索", "查找",  # 语义高度明确，低误匹配风险
            "grep", "import",
            "有没有", "使用了", "调用了", "引用了",
            "搜索代码", "搜索文件", "查找文件", "查找代码",
            "找一下这个", "在哪里用到", "在哪里使用",
            "谁调用了", "被谁引用", "哪里引用",
            "查找引用", "搜索引用", "搜索一下",
            "使用了什么", "调用了什么", "引用了什么",
            "用到哪些", "包含哪些",
        ],
        [
            r"\b(search|find|locate|grep)\b",
            r"\bwhere\s+is\b",
            r"\bwho\s+(calls|uses|references)\b",
            r"\busage\s+of\b",
        ],
    ),
]

# -- 快捷指令关键词（任务复杂度极低）---------------------------------------
# 中文关键词: 子串匹配（不会误匹配）
# 英文关键词: 使用正则单词边界（避免 "hi" 匹配进 "this"、"help" 匹配进 "helper"）

_QUICK_CN: list[str] = [
    "你好", "谢谢", "再见", "帮助", "你能做什么", "你是谁", "版本",
]
"""中文快捷指令 — 子串匹配。"""

_QUICK_EN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bhi\b", re.IGNORECASE),
    re.compile(r"\bhello\b", re.IGNORECASE),
    re.compile(r"\bhey\b", re.IGNORECASE),
    re.compile(r"\bthanks\b", re.IGNORECASE),
    re.compile(r"\bthank you\b", re.IGNORECASE),
    re.compile(r"\bbye\b", re.IGNORECASE),
    re.compile(r"\bhelp\b", re.IGNORECASE),
    re.compile(r"\bwhat can you do\b", re.IGNORECASE),
    re.compile(r"\bwho are you\b", re.IGNORECASE),
    re.compile(r"\bversion\b", re.IGNORECASE),
]
"""英文快捷指令 — 单词边界匹配以避免子串误匹配。"""

# -- 复杂度评估关键词 -------------------------------------------------------

_COMPLEXITY_INDICATORS: dict[str, list[str]] = {
    "complex": [
        "重构", "refactor", "整个项目", "全面分析", "完整分析",
        "所有文件", "所有模块", "全部重构", "全部迁移",
        "迁移", "migrate", "升级", "upgrade", "实现一个",
        "创建一个", "搭建", "从零", "多模块", "跨文件", "多个文件",
        "性能优化", "安全审计", "架构调整", "架构重构", "实施方案",
        "分析和优化", "技术选型", "替代方案",
    ],
    "medium": [
        "修改", "添加", "增加", "实现", "修复", "fix", "bug",
        "测试", "test", "优化", "改进", "改善", "日志", "配置",
        "config", "部署", "deploy", "新建", "创建", "设计",
        "查找所有", "搜索所有", "查找全部", "搜索全部",
    ],
}

# 复杂度评分阈值（score = 长度分 + 关键词分 + 结构分）
# 判定规则:
#   score >= 3  → complex
#   score >= 1  → medium
#   score == 0  → simple


# ============================================================================
# Planner
# ============================================================================


class Planner:
    """任务规划器 — MVP 阶段采用 LLM 隐式规划策略。

    职责：
        - **意图分类** (classify_intent): 本地关键词匹配，零延迟
        - **复杂度评估** (estimate_complexity): 基于输入特征启发式判断
        - **LLM 决策** (think): 调用 LLM 获取初始推理结果

    V2 预留接口（当前抛出 NotImplementedError）：
        - ``plan_explicit()`` — 生成结构化多步执行计划
        - ``revise_plan()`` — 根据工具执行反馈修正计划
        - ``get_plan_progress()`` — 查询计划执行进度

    Attributes:
        llm: LLM 适配器实例，供 ``think()`` 调用。
    """

    # -- V2 计划步骤状态枚举（预留）---------------------------------------
    # V2 将引入 StepStatus 和 PlanStep 数据结构描述
    # 每个步骤的显式计划及其执行状态。

    # -- 类级编译缓存 ----------------------------------------------------
    # 所有 Planner 实例共享相同的正则模式，类级别懒加载避免重复编译。

    _compiled_patterns: ClassVar[dict[str, list[re.Pattern[str]]]] = {}

    @classmethod
    def _get_compiled_patterns(cls) -> dict[str, list[re.Pattern[str]]]:
        """类级懒加载：获取预编译的英文正则模式。

        首次调用时编译并缓存，后续调用直接返回缓存。
        """
        if not cls._compiled_patterns:
            for category, _, eng_patterns in _INTENT_RULES:
                cls._compiled_patterns[category] = [
                    re.compile(p, re.IGNORECASE) for p in eng_patterns
                ]
        return cls._compiled_patterns

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    def __init__(self, llm_adapter: BaseLLMAdapter) -> None:
        """初始化规划器。

        Args:
            llm_adapter: LLM 适配器实例，用于 ``think()`` 中的 LLM 决策调用。
        """
        self.llm = llm_adapter

        logger.info("Planner 初始化完成 (MVP 隐式规划模式)")

    # ------------------------------------------------------------------
    # think — 核心决策方法
    # ------------------------------------------------------------------

    async def think(
        self,
        user_input: str,
        context: ContextPayload,
        tools: list[dict[str, Any]] | None = None,
    ) -> ThoughtResult:
        """分析用户意图并请求 LLM 决策。

        执行步骤：
            1. 本地意图分类 + 复杂度评估（存入日志）
            2. 将上下文消息 + 用户输入发送给 LLM，携带可用工具定义
            3. LLM 返回文本回复 或 工具调用请求 → 封装为 ThoughtResult

        MVP 隐式规划：不生成显式步骤列表，完全依赖 LLM 自身的推理能力
        决定「直接回答」还是「调用工具」。

        .. note::

            MVP 阶段 AgentExecutor 直接调用 LLM 流式接口完成推理+工具调用循环，
            未经过 Planner.think()。本方法预留给 V2 的规划-执行分离架构。
            当前可用于：日志/指标中记录 Planner 的独立决策（与 Executor 对比）。

        Args:
            user_input: 用户输入的文本。
            context: 由 ContextBuilder 构建的完整上下文载荷。
            tools: 可用工具的 OpenAI Function Calling 格式定义列表。
                传入 None 表示本轮不启用工具调用。

        Returns:
            ThoughtResult:
                - ``thought``: LLM 思考/回复文本
                - ``tool_calls``: LLM 请求的工具调用列表
                - ``needs_more_info``: 是否还有工具待执行（True = 有 tool_calls）

        Raises:
            LLMError: LLM API 调用失败时（由 llm.chat() 抛出）。
        """
        # 本地预分析
        intent = self.classify_intent(user_input)
        complexity = self.estimate_complexity(user_input)

        logger.info(
            f"Planner.think() 开始: "
            f"intent={intent}, complexity={complexity}, "
            f"input_len={len(user_input)}"
        )

        # 组装发送给 LLM 的消息列表
        messages = list(context.messages)
        messages.append(Message(role="user", content=user_input))

        # 调用 LLM（携带工具定义以启用 Function Calling）
        logger.info(
            f"Planner.think() 调用 LLM (tools={len(tools) if tools else 0}) ..."
        )
        response = await self.llm.chat(messages, tools=tools)

        # 构造 ThoughtResult
        if response.tool_calls:
            logger.info(
                f"Planner.think() LLM 请求 {len(response.tool_calls)} 个工具调用: "
                f"{[tc.name for tc in response.tool_calls]}"
            )
            result = ThoughtResult(
                thought=response.content,
                tool_calls=response.tool_calls,
                needs_more_info=True,  # 有 tool_calls → 需要执行工具后才能继续
            )
        else:
            logger.info(
                f"Planner.think() LLM 直接回复: "
                f"len={len(response.content or '')}"
            )
            result = ThoughtResult(
                thought=response.content or "",
                tool_calls=None,
                needs_more_info=False,
            )

        return result

    # ------------------------------------------------------------------
    # classify_intent — 本地意图分类
    # ------------------------------------------------------------------

    def classify_intent(self, user_input: str) -> str:
        """快速分类用户意图（基于关键词，纯本地方法，不调用 API）。

        分类优先级（从高到低）:
            1. **快捷指令** — "你好"、"帮助" 等
            2. **意图关键词匹配** — 按 _INTENT_RULES 定义顺序检查
            3. **回退** — 未匹配到任何类别时返回 ``"general_question"``

        设计目标：< 1ms 延迟，零网络开销。用于日志标签、指标统计
        和 V2 显式规划的策略选择。

        Args:
            user_input: 用户输入文本。

        Returns:
            意图分类标签，取以下六种之一:
                - ``"quick_command"`` — 问候 / 帮助 / 版本查询等
                - ``"project_analysis"`` — 项目结构/技术栈分析
                - ``"code_explanation"`` — 代码逻辑解释
                - ``"file_operation"`` — 文件读写/编辑/创建
                - ``"search"`` — 代码搜索/查找引用
                - ``"general_question"`` — 通用技术问答（回退）
        """
        if not user_input or not user_input.strip():
            return "general_question"

        text = user_input.strip()
        text_lower = text.lower()

        # -- 0. 快捷指令优先检测 -----------------------------------------
        # 中文快捷指令：子串匹配
        for cmd in _QUICK_CN:
            if cmd in text:
                logger.debug(f"classify_intent: quick_command (matched cn '{cmd}')")
                return "quick_command"
        # 英文快捷指令：单词边界匹配
        for pattern in _QUICK_EN_PATTERNS:
            if pattern.search(text):
                logger.debug(
                    f"classify_intent: quick_command "
                    f"(matched en '{pattern.pattern}')"
                )
                return "quick_command"

        # -- 1. 关键词得分遍历 -----------------------------------------
        # 使用累计命中数选择最佳匹配类别（而非首个命中即返回）。
        # 这避免了短关键词在早期类别中抢占长关键词在后续类别中的匹配。
        best_category = "general_question"
        best_score = 0

        for category, cn_keywords, _eng_patterns in _INTENT_RULES:
            score = 0

            # 中文关键词子串匹配（每个匹配 +1 分）
            for kw in cn_keywords:
                if kw in text:
                    score += 1
                    logger.debug(
                        f"classify_intent: {category} matched cn '{kw}'"
                    )

            # 英文正则匹配（每个匹配 +2 分，正则更精确）
            for pattern in self._get_compiled_patterns().get(category, []):
                if pattern.search(text):
                    score += 2
                    logger.debug(
                        f"classify_intent: {category} "
                        f"matched en '{pattern.pattern}'"
                    )

            if score > best_score:
                best_score = score
                best_category = category
            # 得分相同时，保留先出现的类别（_INTENT_RULES 顺序即为优先级）

        logger.debug(
            f"classify_intent: -> {best_category} (score={best_score})"
        )
        return best_category

    # ------------------------------------------------------------------
    # estimate_complexity — 本地复杂度评估
    # ------------------------------------------------------------------

    def estimate_complexity(self, user_input: str) -> str:
        """估算任务复杂度（基于启发式规则，纯本地方法）。

        评估维度：
            1. **长度**: 输入越长通常任务越复杂
            2. **关键词**: 匹配 _COMPLEXITY_INDICATORS 中的特征词
            3. **标点计数**: 多个问号/感叹号指示多子任务

        评估结果用于：
            - MVP: 日志记录和性能指标
            - V2:  决定是否启用显式规划（complex 任务制定分步计划）

        Args:
            user_input: 用户输入文本。

        Returns:
            复杂度标签:
                - ``"simple"`` — 单步、直接可回答
                - ``"medium"`` — 需要少量工具调用或分析
                - ``"complex"`` — 多步骤、多文件、需要规划
        """
        if not user_input or not user_input.strip():
            return "simple"

        text = user_input.strip()
        text_lower = text.lower()
        score = 0

        # -- 维度 1: 输入长度 --------------------------------------------
        length = len(text)
        if length > 120:
            score += 4
        elif length > 50:
            score += 1
        # ≤ 50: score += 0

        # -- 维度 2: 关键词特征（取最高命中）----------------------------
        hit_complex = any(kw in text_lower for kw in
                          _COMPLEXITY_INDICATORS.get("complex", []))
        hit_medium = any(kw in text_lower for kw in
                         _COMPLEXITY_INDICATORS.get("medium", []))

        if hit_complex:
            score += 4
        elif hit_medium:
            score += 1

        # -- 维度 3: 多子任务指示 ----------------------------------------
        question_marks = text.count("?") + text.count("？")
        if question_marks >= 3:
            score += 2
        elif question_marks >= 2:
            score += 1

        # 编号列表（"1. ... 2. ..." 或 "1) ... 2) ..."）→ 显式多任务
        if re.search(r"\d[\.\)、]\s", text):
            score += 2

        # -- 判定 --------------------------------------------------------
        if score >= 3:
            return "complex"
        elif score >= 1:
            return "medium"
        else:
            return "simple"

    # ==================================================================
    # V2 预留接口（显式计划）
    # ==================================================================

    async def plan_explicit(
        self,
        user_input: str,
        context: ContextPayload,
    ) -> dict[str, Any]:
        """[V2] 生成结构化的显式多步执行计划。

        V2 中将引入独立的 plan prompt 模板，引导 LLM 输出包含
        步骤标题、描述、预估工具和依赖关系的结构化计划 JSON。

        Args:
            user_input: 用户输入文本。
            context: 上下文载荷。

        Returns:
            dict 包含 ``steps`` (list[PlanStep])、``estimated_iterations``、
            ``risk_notes`` 等字段。

        Raises:
            NotImplementedError: MVP 阶段未实现。
        """
        raise NotImplementedError(
            "plan_explicit() 为 V2 接口，MVP 阶段未实现。"
            "当前请使用 think() 进行隐式规划。"
        )

    async def revise_plan(
        self,
        plan: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> dict[str, Any]:
        """[V2] 根据工具执行反馈动态修正计划。

        当工具执行结果与预期不符时，重新评估计划的可行性，
        根据实际情况调整后续步骤。

        Args:
            plan: 当前执行计划。
            tool_result: 最近一次工具执行的反馈。

        Returns:
            修正后的计划 dict。

        Raises:
            NotImplementedError: MVP 阶段未实现。
        """
        raise NotImplementedError(
            "revise_plan() 为 V2 接口，MVP 阶段未实现。"
        )

    def get_plan_progress(self) -> dict[str, Any] | None:
        """[V2] 查询当前计划的执行进度。

        Returns:
            dict 包含 ``total_steps``、``completed_steps``、
            ``current_step``、``status`` 等字段。
            MVP 阶段始终返回 None。

        Raises:
            NotImplementedError: MVP 阶段未实现。
        """
        raise NotImplementedError(
            "get_plan_progress() 为 V2 接口，MVP 阶段未实现。"
        )
