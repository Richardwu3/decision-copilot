"""
pages/review.py
================
Review Center — 决策复盘中心（st.navigation 多页面应用的一个页面）。

展示该用户全部"已结束比赛"的完整决策复盘：比赛信息、实际比分、
用户预测 vs AI 预测、对比结果（✅/❌）、用户当时的理由、AI 置信度。
按比赛日期降序排列（最近结束的比赛在前）。

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
)


@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    return get_connection()


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

    col1, col2, col3 = st.columns(3)
    col1.metric("已结束比赛", len(review_items))
    col2.metric("已决策", f"{len(decided_items)} / {len(review_items)}")
    if decided_items:
        col3.metric("复盘准确率", f"{correct_count/len(decided_items)*100:.0f}%",
                   help=f"{correct_count}/{len(decided_items)} 场（仅统计你已做出决策的比赛）")
    else:
        col3.metric("复盘准确率", "暂无数据")

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

            with c2:
                st.write("**AI 预测**")
                if pred is not None:
                    # Step 5：与 Match Detail 的 save_decision 写入快照使用同一口径
                    ai_choice, ai_confidence = get_ai_choice_and_confidence(
                        pred["prob_home"], pred["prob_draw"], pred["prob_away"]
                    )
                    ai_choice_cn = choice_label_map.get(ai_choice, ai_choice)

                    ai_correct = is_choice_correct(ai_choice, item["result_code"])
                    ai_icon = "✅" if ai_correct else "❌"

                    st.write(f"选择：{ai_choice_cn} {ai_icon}")
                    st.write(f"置信度：{ai_confidence*100:.1f}%")

                    best_h, best_a, best_p = most_likely_score(pred["lambda_home"], pred["lambda_away"])
                    st.caption(f"预测比分：{best_h}-{best_a} ({best_p*100:.1f}%)")
                else:
                    st.caption("暂无 AI 预测数据。")

            with c3:
                st.write("**实际结果**")
                st.write(f"{item['home_team']} {item['home_goals']} - {item['away_goals']} {item['away_team']}")
                st.write(f"结果：{result_label_map.get(item['result_code'], '未知')}")


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
