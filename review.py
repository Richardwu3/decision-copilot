"""
pages/review.py
================
Review Center — 决策复盘中心（st.navigation 多页面应用的一个页面）。

展示该用户全部"已结束比赛"的完整决策复盘：比赛信息、实际比分、
用户预测 vs AI 预测、对比结果（✅/❌）、用户当时的理由、AI 置信度。
按比赛日期降序排列（最近结束的比赛在前）。

Phase 1 新增：
  - 顶部统计区新增 ROI 指标卡片（用户 / AI模型 / 博彩公司三方对比）。
  - 每场卡片 AI 预测栏新增博彩公司预测结果。
  - AI 预测胜平负逻辑调整：平局概率>30%且主/客胜都<40%则预测平局，否则取最大概率。

数据来源（与 app.py 保持一致，复用同一套判定逻辑，不重复实现）：
  - 用户决策：Google Sheets（database.db.get_user_history），
    经 shared_utils.get_cached_user_history 做 session 内缓存，
    不会因为切换到本页面而重新拉取 Google Sheets（UI修复任务7）。
  - AI 预测、比赛信息、比赛结果：SQLite（database.db_utils）
  - 判断"用户是否选对"、聚合最新决策、AI置信度计算：
    复用 shared_utils.py 中的函数，避免 Review 和 Dashboard / Match Detail
    对"赢/平/负"、"AI置信度"的判定标准出现不一致。

文件位置说明（UI修复任务6）：
  本文件物理路径为 pages/review.py，与 app.py 中
  st.Page("pages/review.py", ...) 的引用路径一致。

身份延续（UI修复任务1）：
  用户昵称的权威来源是 st.session_state（由 shared_utils.get_current_user
  维护），同一浏览器 session 内跨 st.Page 切换不会丢失，不再仅依赖
  st.query_params（st.navigation 切页面时前端是否保留 URL query string
  不受后端代码控制，因此不能作为身份的唯一来源）。
"""

import sqlite3

import streamlit as st

from database.db_utils import (
    get_connection,
    get_all_matches_with_latest_prediction,
    get_match_full_context,
)

# 复用 shared_utils 中已验证过的纯函数，不反向依赖 app.py
from shared_utils import (
    get_current_user,
    render_login_gate,
    build_user_decision_map,
    is_choice_correct,
    most_likely_score,
    get_ai_choice_and_confidence,
    get_cached_user_history,
    compute_value_lab_results,
)

# AI 复盘功能：review.py 只负责渲染按钮与四张卡结果，
# 数据组装（decision_context_builder）与模型调用（ai_coach）
# 全部在各自模块内完成，本文件不直接访问数据库/Google Sheets/Claude API。
from decision_context_builder import build_decision_context
from ai_coach import ClaudeCoach


@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    return get_connection()


@st.cache_resource
def get_ai_coach() -> ClaudeCoach:
    """
    全应用唯一的 ClaudeCoach 实例（@st.cache_resource 保证整个 session
    只初始化一次 client，不必每次点击"AI复盘"都重新读取 API Key）。
    """
    return ClaudeCoach()


# ============================================================
# AI 复盘：按钮 + 四张卡渲染
# ============================================================
#
# 结果按 (user_name, match_id) 缓存在 session_state 里，同一场比赛
# 在本次 session 内只有点击按钮才会真正调用一次 Claude API，页面其他
# 交互触发的 rerun 不会重复消耗 API 调用额度。

_SESSION_KEY_AI_REVIEW_CACHE = "ai_review_cache"


def _get_ai_review_cache() -> dict:
    if _SESSION_KEY_AI_REVIEW_CACHE not in st.session_state:
        st.session_state[_SESSION_KEY_AI_REVIEW_CACHE] = {}
    return st.session_state[_SESSION_KEY_AI_REVIEW_CACHE]


def _render_ai_review_section(user_name: str, match_id: str, match_ctx: dict, decision_record: dict):
    """
    在单场复盘卡片底部渲染"AI复盘"按钮与结果。

    match_ctx / decision_record 直接复用 render_review_center 已经查询好的
    数据（get_match_full_context 的返回值、build_user_decision_map 聚合出的
    该场决策记录），传给 build_decision_context 后不会在 SQLite / Google
    Sheets 上产生任何额外查询。
    """
    cache = _get_ai_review_cache()
    cache_key = (user_name, match_id)

    st.divider()
    btn_label = "🔄 重新生成 AI 复盘" if cache_key in cache else "🤖 生成 AI 复盘"
    clicked = st.button(btn_label, key=f"ai_review_btn_{match_id}")

    if clicked:
        coach = get_ai_coach()
        ctx = build_decision_context(
            user_name=user_name,
            match_id=match_id,
            match_ctx=match_ctx,
            decision_record=decision_record,
        )
        with st.spinner("AI 教练分析中..."):
            review = coach.generate_review(ctx)
        cache[cache_key] = review

    review = cache.get(cache_key)
    if review is None:
        return

    if review.degraded:
        st.info(f"ℹ️ {review.reasoning}")
        return

    score_col, reasoning_col = st.columns([1, 3])
    with score_col:
        st.write("**Decision Score**")
        if review.decision_score is not None:
            st.metric(label="决策评分", value=f"{review.decision_score}/100")
        else:
            st.caption("暂无评分")
        if review.score_rationale:
            st.caption(review.score_rationale)

    with reasoning_col:
        st.write("**Reasoning**")
        st.write(review.reasoning or "（无）")

    bias_col, rec_col = st.columns(2)
    with bias_col:
        st.write("**Bias**")
        if review.biases:
            for b in review.biases:
                st.write(f"- {b}")
        else:
            st.caption("未识别出明显偏差。")
    with rec_col:
        st.write("**Recommendation**")
        if review.recommendations:
            for r in review.recommendations:
                st.write(f"- {r}")
        else:
            st.caption("暂无具体建议。")


# ============================================================
# Phase 1 新增：ROI 与博彩公司预测的辅助函数
# 延迟 import app 中的赔率工具函数，避免模块级循环依赖
# ============================================================

def _get_bookmaker_odds_map() -> dict:
    """从 app.py 延迟导入 load_bookmaker_odds，返回 {match_id(str): odds_raw}。"""
    try:
        from app import load_bookmaker_odds
        return load_bookmaker_odds()
    except Exception:
        return {}


def _american_to_prob(odds: float) -> float:
    """美式赔率 -> 隐含概率（未归一化）。与 app.py 中同名函数逻辑完全一致。"""
    if odds < 0:
        return -odds / (-odds + 100)
    else:
        return 100 / (odds + 100)


def _get_decimal_odds(odds_raw: dict):
    """
    从美式赔率原始数据计算归一化后的十进制赔率。
    返回 (home_dec, draw_dec, away_dec) 或 None（计算失败时）。
    十进制赔率 = 1 / 归一化概率，用于 ROI 计算：
      profit = decimal_odds - 1（下注1单位，猜对时）
    """
    try:
        ph = _american_to_prob(float(odds_raw["odd_win"]))
        pd_ = _american_to_prob(float(odds_raw["odd_draw"]))
        pa = _american_to_prob(float(odds_raw["odd_lose"]))
        total = ph + pd_ + pa
        if total <= 0:
            return None
        return 1 / (ph / total), 1 / (pd_ / total), 1 / (pa / total)
    except Exception:
        return None


def _ai_choice_review(pred) -> str:
    """
    Phase 1：Review Center 使用的 AI 胜平负判断逻辑。
    平局概率 > 30% 且主胜概率 < 40% 且客胜概率 < 40% → 预测平局；
    否则取三者中概率最大的。
    """
    ph = pred["prob_home"]
    pd_ = pred["prob_draw"]
    pa = pred["prob_away"]
    if pd_ > 0.30 and ph < 0.40 and pa < 0.40:
        return "draw"
    prob_map = {"home": ph, "draw": pd_, "away": pa}
    return max(prob_map, key=prob_map.get)


def _bookmaker_choice(odds_raw: dict):
    """
    博彩公司预测：取隐含概率最高（赔率最低）的结果。
    返回 'home' / 'draw' / 'away'，失败返回 None。
    """
    try:
        prob_map = {
            "home": _american_to_prob(float(odds_raw["odd_win"])),
            "draw": _american_to_prob(float(odds_raw["odd_draw"])),
            "away": _american_to_prob(float(odds_raw["odd_lose"])),
        }
        return max(prob_map, key=prob_map.get)
    except Exception:
        return None


def render_review_center(conn: sqlite3.Connection, user_name: str):
    st.title("Review Center")
    st.caption(f"{user_name} 的决策复盘 — 已结束比赛")

    matches = get_all_matches_with_latest_prediction(conn)
    if not matches:
        st.info("暂无比赛数据。请先运行 worldcup_predictor_v4.py 生成预测。")
        return

    user_history = get_cached_user_history(user_name)
    decision_map = build_user_decision_map(user_history)

    # ---- 收集已结束比赛的完整上下文（比赛信息 + AI预测 + 结果）----
    review_items = []
    for m in matches:
        mid = str(m["match_id"])
        ctx = get_match_full_context(conn, m["match_id"])
        result_row = ctx["result"]
        pred_row = ctx["prediction"]

        if result_row is None or not result_row["is_finished"]:
            continue  # Review Center 只展示已结束比赛

        decision = decision_map.get(mid)  # 可能为 None（用户从未对该场决策）

        review_items.append({
            "match_id": mid,
            "date": m["date"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "venue": m["venue"],
            "home_goals": result_row["home_goals"],
            "away_goals": result_row["away_goals"],
            "result_code": result_row["result"],
            "decision": decision,
            "prediction": pred_row,
            # AI复盘用：直接复用这里已经查询好的 get_match_full_context 结果，
            # 传给 decision_context_builder.build_decision_context 时不再
            # 重复查询 SQLite（见该函数 match_ctx 参数说明）。
            "match_ctx": ctx,
        })

    if not review_items:
        st.info("暂无已结束的比赛可供复盘。")
        return

    # ---- 按比赛日期降序排列 ----
    review_items.sort(key=lambda x: x["date"], reverse=True)

    # ---- 顶部汇总：决策覆盖率 + 准确率（仅统计已决策的已结束比赛）----
    decided_items = [r for r in review_items if r["decision"] is not None]
    correct_count = sum(
        1 for r in decided_items
        if is_choice_correct(r["decision"].get("choice"), r["result_code"])
    )

    # ---- Phase 1 新增：ROI 计算 ----
    RESULT_CODE_TO_CHOICE = {"H": "home", "D": "draw", "A": "away"}

    odds_by_match = _get_bookmaker_odds_map()

    user_profit_total = 0.0
    ai_profit_total = 0.0
    book_profit_total = 0.0
    roi_count = 0

    for item in review_items:
        mid = item["match_id"]
        result_code = item["result_code"]
        actual_choice = RESULT_CODE_TO_CHOICE.get(str(result_code).strip().upper())
        if actual_choice is None:
            continue

        odds_raw = odds_by_match.get(mid)
        if not odds_raw:
            continue  # 无赔率数据，跳过该场

        dec_odds = _get_decimal_odds(odds_raw)
        if dec_odds is None:
            continue

        home_dec, draw_dec, away_dec = dec_odds
        odds_for_choice = {"home": home_dec, "draw": draw_dec, "away": away_dec}

        roi_count += 1

        # 用户 ROI（只统计已决策场次，未决策视为未下注，不纳入该用户 ROI 分母）
        user_decision = item["decision"]
        if user_decision and user_decision.get("choice") in ("home", "draw", "away"):
            user_c = user_decision["choice"]
            user_profit_total += (odds_for_choice[user_c] - 1) if user_c == actual_choice else -1

        # AI 模型 ROI（每场都参与）
        pred = item["prediction"]
        if pred is not None:
            ai_c = _ai_choice_review(pred)
            ai_profit_total += (odds_for_choice[ai_c] - 1) if ai_c == actual_choice else -1

        # 博彩公司 ROI（每场都参与）
        book_c = _bookmaker_choice(odds_raw)
        if book_c:
            book_profit_total += (odds_for_choice[book_c] - 1) if book_c == actual_choice else -1

    # ---- 展示：顶部统计区（保持原有三列，ROI 独立一行）----
    col1, col2, col3 = st.columns(3)
    col1.metric("已结束比赛", len(review_items))
    col2.metric("已决策", f"{len(decided_items)} / {len(review_items)}")
    if decided_items:
        col3.metric("复盘准确率", f"{correct_count/len(decided_items)*100:.0f}%",
                   help=f"{correct_count}/{len(decided_items)} 场（仅统计你已做出决策的比赛）")
    else:
        col3.metric("复盘准确率", "暂无数据")

    # ---- Phase 1 新增：ROI 指标卡片 ----
    if roi_count > 0:
        def _fmt_roi(val: float) -> str:
            return f"{'+' if val >= 0 else ''}{val:.0f}%"

        # 用户 ROI 分母只计算已决策的场次
        user_decided_roi_count = sum(
            1 for item in review_items
            if item["decision"] and item["decision"].get("choice") in ("home", "draw", "away")
            and odds_by_match.get(item["match_id"])
            and _get_decimal_odds(odds_by_match[item["match_id"]]) is not None
        )
        user_roi_str = (_fmt_roi(user_profit_total / user_decided_roi_count * 100)
                        if user_decided_roi_count > 0 else "暂无")
        ai_roi_str   = _fmt_roi(ai_profit_total   / roi_count * 100)
        book_roi_str = _fmt_roi(book_profit_total / roi_count * 100)

        st.metric(
            "ROI（每场固定下注1单位）",
            f"您: {user_roi_str}　｜　AI模型: {ai_roi_str}　｜　博彩公司: {book_roi_str}",
            help=(f"AI模型 & 博彩公司基于 {roi_count} 场有赔率数据的已结束比赛；"
                  f"您的ROI仅统计其中您已决策的 {user_decided_roi_count} 场"),
        )
    else:
        st.metric("ROI", "暂无数据", help="需要至少一场有赔率数据的已结束比赛")

    # ---- 任务3新增：Value Bet Snapshot 摘要卡片 ----
    # 复用 shared_utils.compute_value_lab_results（与 Value Lab 页面同一套
    # 纯函数计算逻辑），保证这里展示的 ROI 与 Value Lab 页面完全一致，
    # 这里只展示摘要，完整下注明细见 Value Lab 页面。
    st.divider()
    st.subheader("Value Bet Snapshot")

    value_lab_data = compute_value_lab_results(user_name, conn)
    vl_eligible_count = value_lab_data["eligible_count"]

    if vl_eligible_count == 0:
        st.caption("暂无可回测的比赛（需要至少一场已结束、且决策时保存了完整AI/市场/赔率快照的比赛）。")
    else:
        st.caption(f"基于 {vl_eligible_count} 场有快照的比赛")
        for key in ("A", "B", "C"):
            r = value_lab_data["strategies"][key]
            if r["roi"] is not None:
                sign = "+" if r["roi"] >= 0 else ""
                roi_str = f"{sign}{r['roi']:.0f}%"
            else:
                roi_str = "暂无"
            st.write(f"{r['label']}：{roi_str} ROI（{r['bet_count']} bets）")

    st.page_link("value_lab.py", label="查看完整策略分析 →", icon="📊")

    st.divider()

    # ---- 逐场复盘卡片 ----
    result_label_map = {"H": "主胜", "D": "平局", "A": "客胜"}
    choice_label_map = {"home": "主胜", "draw": "平局", "away": "客胜"}

    for item in review_items:
        decision = item["decision"]
        pred = item["prediction"]

        with st.container(border=True):
            # ---- 比赛信息 + 实际比分 ----
            header_col, badge_col = st.columns([4, 1])
            with header_col:
                st.subheader(f"{item['home_team']} {item['home_goals']} - {item['away_goals']} {item['away_team']}")
                st.caption(f"{item['date']} · {item['venue'] or '场地待定'} · "
                          f"结果：{result_label_map.get(item['result_code'], '未知')}")

            with badge_col:
                if decision is None:
                    st.warning("⚪ 未决策")
                else:
                    correct = is_choice_correct(decision.get("choice"), item["result_code"])
                    if correct:
                        st.success("✅ 判断正确")
                    else:
                        st.error("❌ 判断错误")

            # ---- 三列对比：用户预测 / AI预测 / 实际结果 ----
            c1, c2, c3 = st.columns(3)

            with c1:
                st.write("**你的预测**")
                if decision is None:
                    st.caption("你没有对这场比赛做出决策。")
                else:
                    choice_cn = choice_label_map.get(decision.get("choice"), decision.get("choice"))
                    st.write(f"选择：{choice_cn}")
                    if decision.get("confidence"):
                        st.write(f"信心：{decision.get('confidence')}/5")
                    if decision.get("reason"):
                        st.caption(f"理由：{decision.get('reason')}")
                    # 任务2：显示 comment（安全读取，旧数据缺失该字段时不报错）
                    if decision.get("comment", ""):
                        st.caption(f"备注：{decision.get('comment', '')}")

            with c2:
                st.write("**AI 预测**")
                if pred is not None:
                    # Phase 1：使用新的 AI 选择逻辑（平局>30%且主/客<40%→平局）
                    ai_c = _ai_choice_review(pred)
                    ai_choice_cn = choice_label_map.get(ai_c, ai_c)
                    ai_correct = is_choice_correct(ai_c, item["result_code"])
                    ai_icon = "✅" if ai_correct else "❌"
                    st.write(f"选择：{ai_choice_cn} {ai_icon}")

                    # 置信度仍用 get_ai_choice_and_confidence（取最大概率值）
                    _, ai_conf = get_ai_choice_and_confidence(
                        pred["prob_home"], pred["prob_draw"], pred["prob_away"]
                    )
                    st.write(f"置信度：{ai_conf*100:.1f}%")

                    best_h, best_a, best_p = most_likely_score(pred["lambda_home"], pred["lambda_away"])
                    st.caption(f"预测比分：{best_h}-{best_a} ({best_p*100:.1f}%)")

                    # Phase 1 新增：博彩公司预测
                    odds_raw_item = odds_by_match.get(item["match_id"])
                    if odds_raw_item:
                        book_c = _bookmaker_choice(odds_raw_item)
                        if book_c:
                            book_correct = is_choice_correct(book_c, item["result_code"])
                            book_icon = "✅" if book_correct else "❌"
                            st.caption(f"博彩公司预测：{choice_label_map.get(book_c, book_c)} {book_icon}")
                else:
                    st.caption("暂无 AI 预测数据。")

            with c3:
                st.write("**实际结果**")
                st.write(f"{item['home_team']} {item['home_goals']} - {item['away_goals']} {item['away_team']}")
                st.write(f"结果：{result_label_map.get(item['result_code'], '未知')}")

            # ---- AI 复盘（仅对已决策的比赛可用；未决策则无需展示按钮）----
            if decision is not None:
                _render_ai_review_section(
                    user_name=user_name,
                    match_id=item["match_id"],
                    match_ctx=item["match_ctx"],
                    decision_record=decision,
                )


def main():
    user_name = get_current_user()
    if not user_name:
        render_login_gate()
        return

    conn = get_db_connection()

    with st.sidebar:
        st.header("Decision Review Copilot")
        st.caption(f"当前用户：{user_name}")

    render_review_center(conn, user_name)


# 注意：不使用 `if __name__ == "__main__":` 来判断是否执行 main()。
# Streamlit 在用 st.Page("pages/review.py", ...) 这种文件路径方式调度页面时，
# 会把本文件作为脚本 exec 执行，并将其 __name__ 强制设为 "__main__"
# （这是 Streamlit 源码中的官方行为，注释明确说明此模式下不能依赖 __name__ 判断），
# 因此直接调用 main()，不加条件判断，语义更准确。
main()
