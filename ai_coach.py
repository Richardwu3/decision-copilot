"""
ai_coach.py
============
AI 教练模块 —— 基于 decision_context_builder 输出的 DecisionContext，
调用 Claude API 生成结构化的决策复盘（四张卡：Decision Score / Reasoning /
Bias / Recommendation）。

硬约束：
  - 只评价决策质量，不评价结果（SYSTEM_PROMPT 中明确禁止"运气好/运气不好"
    式的结果导向表述）。
  - Prompt 拆分为独立变量（SYSTEM_PROMPT / USER_PROMPT_TEMPLATE），
    方便后续迭代版本时直接改字符串，不需要改代码结构。
  - API 不可用（未安装 SDK / 无 key / 网络失败 / 返回格式异常）时优雅降级，
    返回 CoachReview(degraded=True, ...)，不抛异常中断 Streamlit 页面渲染。
  - 本文件不直接访问数据库或 Google Sheets —— 只接受
    decision_context_builder.DecisionContext 作为输入，保持
    "decision_context_builder 是唯一数据入口"这一硬约束。
"""

import os
import json
import re
from dataclasses import dataclass
from typing import Optional, List

try:
    import anthropic
    _ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    _ANTHROPIC_SDK_AVAILABLE = False

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


# 可按需切换为其他 Claude 模型，只改这一个常量。
CLAUDE_MODEL = "claude-sonnet-4-5"


# ============================================================
# Prompt（独立变量，便于版本迭代；改动这里不需要碰任何其他代码）
# ============================================================

SYSTEM_PROMPT = """你是 Decision Copilot 的 AI 足球决策教练。你的唯一职责是评价用户"决策过程"的质量，
而不是评价"结果好坏"。

硬性原则：
1. 绝对不能出现"你运气不好/运气好"这类结果导向的表述。即使比赛结果与用户预测不符，
   只要决策过程本身逻辑自洽、依据充分，就应该给出正面评价；反之，即使结果蒙对了，
   如果决策依据薄弱（例如纯凭直觉、信息不足、与模型和市场都严重背离且无合理理由），
   也应该指出问题。
2. 只依据我提供给你的结构化数据做分析，不要编造未提供的信息（伤病、战绩、球员状态等）。
3. 输出必须是严格的 JSON，不要有任何 JSON 之外的文字，不要用 Markdown 代码块包裹。
4. 语言：使用简体中文，语气专业、直接、有建设性，避免空泛的套话。

输出 JSON 结构（对应四张卡）：
{
  "decision_score": <0-100的整数，决策过程质量评分，不是结果评分>,
  "score_rationale": "<一句话说明评分依据>",
  "reasoning": "<2-4句话，分析用户这次决策的逻辑：依据是否清晰、与AI/市场的关系如何、
                信心水平是否匹配依据强度>",
  "biases": ["<识别出的认知偏差或决策模式，每条一句话；没有则给空数组>"],
  "recommendations": ["<可执行的改进建议，每条一句话，2-4条>"]
}
"""

USER_PROMPT_TEMPLATE = """请基于以下决策快照，生成一次结构化复盘。

【比赛信息】
{match_line}

【用户决策】
{user_decision_block}

【AI 模型观点（决策时刻快照）】
{ai_block}

【市场观点（决策时刻快照）】
{market_block}

【EV 快照】
{ev_block}

【决策标签（规则引擎自动标注，供你参考决策所处的情境，不代表结论）】
{taxonomy_line}

【实际赛果】
{result_block}

请严格按 SYSTEM_PROMPT 中定义的 JSON 结构输出，不要输出任何其他内容。
"""


# ============================================================
# 输出数据结构
# ============================================================

@dataclass
class CoachReview:
    decision_score: Optional[int] = None
    score_rationale: str = ""
    reasoning: str = ""
    biases: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    degraded: bool = False        # True 表示这是降级结果，不是真实 AI 输出
    degraded_reason: str = ""

    def __post_init__(self):
        if self.biases is None:
            self.biases = []
        if self.recommendations is None:
            self.recommendations = []


# ============================================================
# Prompt 组装辅助函数
# ============================================================

def _fmt_pct(v):
    return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "数据缺失"


def _fmt_choice(c):
    return {"home": "主胜", "draw": "平局", "away": "客胜"}.get(c, c or "未知")


def _build_user_prompt(ctx) -> str:
    m = ctx.match
    ud = ctx.user_decision
    ai = ctx.ai_snapshot
    mk = ctx.market_snapshot
    ev = ctx.ev_snapshot
    res = ctx.result

    match_line = f"{m.home_team or '主队'} vs {m.away_team or '客队'}（{m.date or '日期未知'}，阶段：{m.stage or '未知'}）"

    if ud.has_decision:
        user_decision_block = (
            f"选择：{_fmt_choice(ud.choice)}　信心：{ud.confidence if ud.confidence is not None else '未知'}/5\n"
            f"比分预测：{ud.predict_home if ud.predict_home is not None else '?'}"
            f"-{ud.predict_away if ud.predict_away is not None else '?'}\n"
            f"理由标签：{ud.reason or '未填写'}\n"
            f"备注：{ud.comment or '无'}"
        )
    else:
        user_decision_block = "用户尚未对该场比赛做出决策。"

    if ai.available:
        ai_block = (
            f"主胜 {_fmt_pct(ai.prob_home)} · 平局 {_fmt_pct(ai.prob_draw)} · 客胜 {_fmt_pct(ai.prob_away)}\n"
            f"AI 选择：{_fmt_choice(ai.ai_choice)}（置信度 {_fmt_pct(ai.ai_confidence)}）"
        )
    else:
        ai_block = "AI 快照数据缺失。"

    if mk.available:
        market_block = (
            f"主胜 {_fmt_pct(mk.prob_home)} · 平局 {_fmt_pct(mk.prob_draw)} · 客胜 {_fmt_pct(mk.prob_away)}\n"
            f"市场选择：{_fmt_choice(mk.market_choice)}"
        )
    else:
        market_block = "市场快照数据缺失。"

    if ev.available:
        ev_lines = (
            f"主胜EV {ev.ev_home:+.1%} · 平局EV {ev.ev_draw:+.1%} · 客胜EV {ev.ev_away:+.1%}"
        )
        if ev.ev_for_user_choice is not None:
            ev_lines += f"\n用户所选结果的EV：{ev.ev_for_user_choice:+.1%}"
        ev_block = ev_lines
    else:
        ev_block = "EV 快照数据缺失。"

    taxonomy_line = "、".join(ctx.decision_taxonomy) if ctx.decision_taxonomy else "无标签"

    if res.is_finished:
        result_block = f"{m.home_team} {res.home_goals} - {res.away_goals} {m.away_team}"
    else:
        result_block = "比赛尚未结束（或无结果数据）。"

    return USER_PROMPT_TEMPLATE.format(
        match_line=match_line,
        user_decision_block=user_decision_block,
        ai_block=ai_block,
        market_block=market_block,
        ev_block=ev_block,
        taxonomy_line=taxonomy_line,
        result_block=result_block,
    )


# ============================================================
# JSON 解析（容错：模型偶尔仍会包一层 ```json 代码块）
# ============================================================

def _parse_review_json(text: str) -> Optional[dict]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


# ============================================================
# ClaudeCoach
# ============================================================

class ClaudeCoach:
    """
    AI 决策教练。generate_review() 是当前唯一实现的方法；
    generate_profile() / generate_pre_match_alert() 预留为未来扩展点，
    复用同一个 client 初始化与降级逻辑，不需要重新实现一遍 API 调用细节。
    """

    def __init__(self, api_key: Optional[str] = None, model: str = CLAUDE_MODEL):
        """
        api_key 优先级：
          1. 显式传入的 api_key 参数
          2. st.secrets["ANTHROPIC_API_KEY"]（Streamlit Cloud 部署场景）
          3. 环境变量 ANTHROPIC_API_KEY（本地开发场景，`export ANTHROPIC_API_KEY=sk-ant-xxxx`）

        故意不在代码里硬编码任何真实 key 字面量：即使项目里已经有一个可用
        的 key，密钥也应该只存在于 secrets / 环境变量里，不进代码仓库。
        本地运行时把你的 key 写进 .streamlit/secrets.toml 或 export 成环境
        变量即可，本类会自动读取，不需要改这里的代码。
        """
        self.model = model
        self.api_key = api_key or self._resolve_api_key()
        self.system_prompt = SYSTEM_PROMPT
        self.client = None

        if self.api_key and _ANTHROPIC_SDK_AVAILABLE:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                print(f"[ai_coach.py] 警告：初始化 Anthropic client 失败（{e}），AI复盘功能将降级。")
                self.client = None

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        if _STREAMLIT_AVAILABLE:
            try:
                if "ANTHROPIC_API_KEY" in st.secrets:
                    return st.secrets["ANTHROPIC_API_KEY"]
            except Exception:
                pass
        return os.environ.get("ANTHROPIC_API_KEY")

    def _degraded(self, reason: str) -> CoachReview:
        return CoachReview(
            decision_score=None,
            score_rationale="",
            reasoning=f"AI复盘暂不可用：{reason}",
            biases=[],
            recommendations=[],
            degraded=True,
            degraded_reason=reason,
        )

    def generate_review(self, decision_context, max_tokens: int = 1000) -> CoachReview:
        """
        输入：decision_context_builder.DecisionContext
        输出：CoachReview（四张卡的数据）。

        优雅降级场景（均不抛异常）：
          - 用户尚未做出决策 -> 直接降级，不浪费一次 API 调用
          - SDK 未安装 / 无 API key -> degraded=True，说明原因
          - API 调用异常（网络/超时/限流）-> degraded=True
          - 返回内容无法解析为 JSON -> degraded=True，原始文本附在
            reasoning 里供排查（截断至500字符，避免污染UI）
        """
        if not decision_context.user_decision.has_decision:
            return self._degraded("该场比赛用户尚未做出决策，无法生成复盘。")

        if self.client is None:
            reason = "未配置有效的 Anthropic API Key" if not self.api_key else "anthropic SDK 未安装（pip install anthropic）"
            return self._degraded(reason)

        user_prompt = _build_user_prompt(decision_context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            raw_text = "\n".join(text_parts).strip()
        except Exception as e:
            return self._degraded(f"API 调用失败（{type(e).__name__}: {e}）")

        parsed = _parse_review_json(raw_text)
        if parsed is None:
            review = self._degraded("模型返回内容无法解析为预期格式。")
            review.reasoning += f"\n\n原始返回内容：\n{raw_text[:500]}"
            return review

        try:
            score = parsed.get("decision_score")
            score = int(score) if score is not None else None
        except (TypeError, ValueError):
            score = None

        return CoachReview(
            decision_score=score,
            score_rationale=str(parsed.get("score_rationale", "")),
            reasoning=str(parsed.get("reasoning", "")),
            biases=list(parsed.get("biases", []) or []),
            recommendations=list(parsed.get("recommendations", []) or []),
            degraded=False,
        )

    # ---- 预留扩展点（本次不实现，仅占位说明，复用 self.client / self.model） ----
    # def generate_profile(self, decision_contexts: list) -> "ProfileReview":
    #     """输入 decision_context_builder.build_decision_context_batch() 的结果，
    #     生成用户决策画像（例如：擅长/薄弱的比赛类型、系统性偏差）。"""
    #     ...
    #
    # def generate_pre_match_alert(self, decision_context) -> "AlertReview":
    #     """输入尚未决策的比赛的 DecisionContext（user_decision.has_decision=False），
    #     生成赛前提醒（例如：模型与市场分歧较大，建议重点关注）。"""
    #     ...
