"""
value_lab.py
================
Value Lab / 策略实验室 —— st.navigation 多页面应用的一个页面。

页面定位：回答一个问题——如果完全按照模型给出的 EV 下注，长期会发生什么？

数据来源规则（非常重要，必须严格遵守，实现见 shared_utils.compute_value_lab_results）：
  1. Google Sheet 是 Value Lab 的唯一真相来源：所有策略回测使用决策时刻
     保存的快照字段（ai_*_prob, market_*_prob, home/draw/away_odds, ev_*），
     不重新从 SQLite 当前预测结果计算 AI 概率。
  2. SQLite 只用于补充比赛名称、时间、场地和实际赛果（result）。
  3. 同一用户对同一 match_id 的多条记录，只取最新 timestamp 一条。
  4. 旧数据兼容：
       - 缺失 market_*_prob 或原始赔率 -> 跳过该场
       - 缺失 ev_* 但 ai_*_prob 和 market_*_prob 都在 -> 临时补算 EV
       - ai_*_prob 也缺失 -> 跳过该场
  5. 只统计已结束且有完整快照的比赛。

三条策略定义：
  策略A：Top EV > 5%          —— 取EV最高的结果，EV>5%则下注，否则跳过。
  策略B：Top EV + 模型主选一致 —— EV最高的结果必须同时是AI模型主选（概率最大者），且EV>5%。
  策略C：高置信度 + 正EV       —— AI模型主选概率≥50%，且该结果EV>5%。

盈亏计算：下注1单位；猜中 profit = 十进制赔率 - 1；猜错 profit = -1。
十进制赔率 = 1 / 决策时刻保存的（归一化后）市场隐含概率。

本文件与 review.py 遵循同一套 st.navigation 页面调度约定：不使用
`if __name__ == "__main__":` 判断，文件末尾无条件调用 main()
（原因见 review.py 文件末尾的注释——Streamlit 用 st.Page(文件路径) 方式
调度页面时会把本文件当独立脚本 exec 执行）。

核心计算逻辑（构建可回测比赛列表、三条策略、回测模拟）全部实现在
shared_utils.py 中的纯函数里（不含任何 st.* 调用），本文件只负责渲染，
review.py 的「Value Bet Snapshot」摘要卡片复用同一套计算函数，
两处展示的数字保证同一口径、不重复实现。
"""

import sqlite3

import streamlit as st

from database.db_utils import get_connection

from shared_utils import (
    get_current_user,
    render_login_gate,
    compute_value_lab_results,
    compute_strategy_a_diagnostics,
)


@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    return get_connection()


CHOICE_LABEL = {"home": "主胜", "draw": "平局", "away": "客胜"}


def _fmt_signed_pct(value) -> str:
    """把 12.3 格式化为 '+12%'，None 格式化为 '暂无'。"""
    if value is None:
        return "暂无"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.0f}%"


def _render_strategy_card(col, result: dict):
    """渲染顶部策略总览卡中的一张卡片（策略A/B/C 或 你的真人决策）。"""
    with col:
        with st.container(border=True):
            st.markdown(f"**{result['label']}**")
            st.metric("ROI", _fmt_signed_pct(result["roi"]))
            if result["hit_rate"] is not None:
                st.caption(f"命中率：{result['hit_rate']*100:.0f}%")
            else:
                st.caption("命中率：暂无")
            st.caption(f"下注场次：{result['bet_count']}")
            if result["avg_ev"] is not None:
                st.caption(f"平均EV：{result['avg_ev']*100:+.1f}%")
            else:
                st.caption("平均EV：暂无")
            st.caption(f"最大回撤：{result['max_drawdown']:.2f} 单位")


def render_value_lab(conn: sqlite3.Connection, user_name: str):
    st.title("Value Lab / 策略实验室")
    st.caption("如果完全按照模型给出的 EV 下注，长期会发生什么？")

    data = compute_value_lab_results(user_name, conn)
    eligible_count = data["eligible_count"]

    if eligible_count == 0:
        st.info(
            "暂无可回测的比赛。Value Lab 需要至少一场已结束、且你在 Match Detail "
            "提交决策时保存了完整 AI/市场概率与赔率快照的比赛。"
        )
        return

    st.caption(f"基于 {eligible_count} 场有完整快照的已结束比赛")

    # ---- 策略总览卡（4个卡片并列）----
    col_a, col_b, col_c, col_real = st.columns(4)
    _render_strategy_card(col_a, data["strategies"]["A"])
    _render_strategy_card(col_b, data["strategies"]["B"])
    _render_strategy_card(col_c, data["strategies"]["C"])
    _render_strategy_card(col_real, data["real"])

    st.divider()

    # ---- 策略定义说明 ----
    with st.expander("策略定义说明", expanded=False):
        st.markdown(
            "- **策略A：Top EV > 5%** —— 对每场比赛取三个结果中EV最高的一个，"
            "若该EV超过5%则下注1单位，否则跳过。\n"
            "- **策略B：Top EV + 模型主选一致** —— 仅当EV最高的结果同时也是"
            "AI模型的主选（三个结果中概率最大者），且EV超过5%时才下注，否则跳过。\n"
            "- **策略C：高置信度 + 正EV** —— 仅当AI模型主选的概率达到或超过50%，"
            "且该结果的EV超过5%时才下注，否则跳过。\n\n"
            "三条策略均下注1单位；命中时盈利 = 十进制赔率 - 1，未命中时亏损1单位。"
        )

    st.divider()

    # ---- 策略对比表 ----
    st.subheader("策略对比")
    comparison_rows = []
    for key in ("A", "B", "C"):
        r = data["strategies"][key]
        comparison_rows.append({
            "策略": r["label"],
            "下注场次": r["bet_count"],
            "命中率": f"{r['hit_rate']*100:.0f}%" if r["hit_rate"] is not None else "—",
            "ROI": _fmt_signed_pct(r["roi"]),
            "平均EV": f"{r['avg_ev']*100:+.1f}%" if r["avg_ev"] is not None else "—",
            "最大回撤": f"{r['max_drawdown']:.2f}",
        })
    st.table(comparison_rows)

    st.divider()

    # ---- 下注明细表 ----
    st.subheader("下注明细")
    strategy_key = st.selectbox(
        "选择策略查看下注明细",
        options=["A", "B", "C"],
        format_func=lambda k: data["strategies"][k]["label"],
    )
    bets = data["strategies"][strategy_key]["bets"]

    if not bets:
        st.caption("该策略在当前数据下没有产生任何下注。")
    else:
        detail_rows = []
        for b in bets:
            detail_rows.append({
                "比赛": f"{b['home_team']} vs {b['away_team']} ({b['date']})",
                "下注结果": CHOICE_LABEL.get(b["outcome"], b["outcome"]),
                "EV": f"{b['ev']*100:+.0f}%",
                "赔率": f"{b['odds']:.2f}",
                "赛果": CHOICE_LABEL.get(b["actual_choice"], b["actual_choice"]),
                "盈亏": f"{'+' if b['profit'] >= 0 else ''}{b['profit']:.2f}",
            })
        st.dataframe(detail_rows, width='stretch', hide_index=True)

    st.divider()
    _render_diagnostics(user_name, conn)


# ============================================================
# Diagnostics / 策略A诊断 —— 渲染层
# 所有聚合计算都在 shared_utils.compute_strategy_a_diagnostics 完成，
# 本文件只负责格式化数值与用 st.tabs/st.table/st.info/st.warning 展示。
# ============================================================

def _fmt_frac_pct(value, signed: bool = False) -> str:
    """把 0~1（或更大）的小数格式化为百分比字符串，None 显示为 '—'。"""
    if value is None:
        return "—"
    pct = value * 100
    sign = "+" if signed and pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _fmt_roi_pct(value) -> str:
    """ROI 值本身已经是百分比数值（如 12.3 表示 +12.3%）。"""
    if value is None:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _fmt_num(value, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _render_diagnostic_table(table: dict, columns_fmt: dict):
    """
    渲染单张诊断表 + 自动结论。
    columns_fmt: {列名: 格式化函数}，未在字典中出现的列按原样展示（如“下注类型”“赔率区间”这类字符串列、“场次”这类整数列）。
    """
    rows = table["rows"]
    if not rows:
        st.caption("暂无数据（该分组下没有下注记录，可能是样本量太少或全部字段缺失被跳过）。")
        return

    display_rows = []
    for r in rows:
        display_row = {}
        for k, v in r.items():
            fmt = columns_fmt.get(k)
            display_row[k] = fmt(v) if fmt else v
        display_rows.append(display_row)
    st.table(display_rows)

    if table.get("insufficient_sample"):
        st.caption(f"⚠️ 相关分组下注场次较少，结论仅供参考。")

    conclusion = table.get("conclusion")
    if conclusion:
        st.warning(f"📌 {conclusion}")
    else:
        st.info("暂未发现明显的系统性偏差。")


def _render_diagnostics(user_name: str, conn: sqlite3.Connection):
    st.subheader("🔍 Diagnostics / 策略A诊断")

    diag = compute_strategy_a_diagnostics(user_name, conn)
    n_bets = diag["n_bets"]

    if n_bets == 0:
        st.caption("策略A（Top EV>5%）在当前数据下没有产生任何下注，暂无诊断数据。")
        return

    st.caption(f"基于策略A（Top EV>5%）的 {n_bets} 场下注记录分析")

    tab1, tab2, tab3, tab4 = st.tabs([
        "按主/平/客拆分", "按赔率区间", "按模型-市场分歧", "按Elo差距",
    ])

    with tab1:
        _render_diagnostic_table(
            diag["tables"]["by_bet_type"],
            columns_fmt={
                "命中率": _fmt_frac_pct,
                "ROI": _fmt_roi_pct,
                "平均EV": lambda v: _fmt_frac_pct(v, signed=True),
                "平均赔率": lambda v: _fmt_num(v, 2),
                "最大回撤": lambda v: _fmt_num(v, 2),
            },
        )

    with tab2:
        _render_diagnostic_table(
            diag["tables"]["by_odds_bucket"],
            columns_fmt={
                "命中率": _fmt_frac_pct,
                "ROI": _fmt_roi_pct,
                "平均EV": lambda v: _fmt_frac_pct(v, signed=True),
                "市场隐含胜率": _fmt_frac_pct,
                "模型平均预测概率": _fmt_frac_pct,
            },
        )

    with tab3:
        _render_diagnostic_table(
            diag["tables"]["by_gap_bucket"],
            columns_fmt={
                "命中率": _fmt_frac_pct,
                "ROI": _fmt_roi_pct,
                "平均EV": lambda v: _fmt_frac_pct(v, signed=True),
                "平局下注占比": _fmt_frac_pct,
                "冷门赔率占比": _fmt_frac_pct,
            },
        )

    with tab4:
        if not diag["elo_available"]:
            st.info(
                "暂无可用的 Elo 数据（SQLite matches 表中查不到这批比赛的 "
                "home_elo / away_elo），此项诊断暂不可用，不影响其他3张诊断表。"
            )
        else:
            _render_diagnostic_table(
                diag["tables"]["by_elo_bucket"],
                columns_fmt={
                    "命中率": _fmt_frac_pct,
                    "ROI": _fmt_roi_pct,
                    "平局EV占比": _fmt_frac_pct,
                    "模型-市场平均分歧": _fmt_frac_pct,
                },
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

    render_value_lab(conn, user_name)


# 注意：与 review.py 一致，不使用 `if __name__ == "__main__":` 判断，
# 直接调用 main()（原因见 review.py 文件末尾注释：st.Page(文件路径)
# 调度方式下 Streamlit 会把本文件当独立脚本执行）。
main()
