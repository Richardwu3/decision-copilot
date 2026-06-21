"""
app.py
======
AI-Powered Decision Review Copilot — Phase 1 MVP（Step 3：Match Detail 三栏对比迁移）

包含两个视图：
  1. Dashboard    — "今天我需要做哪些决策？"（Step 2 已迁移到 Google Sheets）
  2. Match Detail — 决策详情页，Step 3 起改为三栏对比布局：
       左栏：用户预测（选择、信心、理由）—— 来自 Google Sheets
       中栏：AI 预测（WDL 概率、λ、最可能比分）—— 来自 SQLite predictions 表
       右栏：博彩公司预测（赔率转概率）—— 来自 schedule_2026_result.xls

Step 3 变更：
  - render_match_detail 完全迁移到 Google Sheets：写入决策用 database.db.save_decision，
    读取决策历史用 database.db.get_user_history，不再使用 SQLite 的
    decision_events 表（save_decision_event/get_latest_decision/get_decision_history
    已从 import 中移除）。
  - 过渡常量 USER_NAME 已删除，render_match_detail 现在接收 user_name 参数，
    与 Dashboard 共享同一个通过 st.query_params 识别的用户身份。
  - 新增博彩赔率读取（schedule_2026_result.xls）与伤病球员姓名读取
    （Injury_report.xlsx），均为只读，不写入任何数据库。
  - 特征解释简化为只展示 Top Positive 和 Top Negative（各取贡献度最大的一项）。
  - 比赛/预测/特征/结果数据仍从 SQLite 读取，这部分留待后续 Step 处理。

运行方式：
    streamlit run app.py

前置条件：
    1. 已运行 python database/init_db.py 初始化数据库
    2. 已运行 python worldcup_predictor_v4.py 生成预测数据 + model_coefficients.json
    3. 已配置 credentials.json，且 Google Sheet "Decision_Logs" 已与 service account
       邮箱共享编辑权限（Google Sheets 不可用时，决策记录功能优雅降级为空）
    4. data/schedule_2026_result.xls 与 data/Injury_report.xlsx 位于项目 data/ 目录
       （缺失时，对应栏位会提示"暂无数据"而不是报错）
"""

import os
import json
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

from database.db_utils import (
    get_connection,
    ensure_database_schema,
    get_all_matches_with_latest_prediction,
    get_match_full_context,
)
from database.db import (
    save_decision,
    get_all_decisions,
)
from shared_utils import (
    get_current_user,
    set_current_user,
    render_login_gate,
    is_choice_correct,
    build_user_decision_map,
    most_likely_score,
    get_ai_choice_and_confidence,
    get_cached_user_history,
    invalidate_user_history_cache,
    REASON_TAG_OPTIONS,
)

COEF_PATH = "model_coefficients.json"

# 人类可读的特征名称（仅用于展示，不影响计算）
FEATURE_DISPLAY_NAMES = {
    "elo_diff": "Elo Difference",
    "log_value_ratio": "Market Value",
    "weighted_xg_diff": "Attacking Form (xG)",
    "weighted_xga_diff": "Defensive Form (xGA)",
    "home_adv": "Home Advantage",
}

# REASON_TAG_OPTIONS 已迁移至 shared_utils.py（任务4：恢复决策理由下拉菜单）

WORLDCUP_RESULT_PATH = "data/schedule_2026_result.xls"
INJURY_REPORT_PATH = "data/Injury_report.xlsx"


# ============================================================
# ============================================================
# 用户身份函数（get_current_user / set_current_user / render_login_gate）
# 已迁移至 shared_utils.py，供 app.py 与 pages/review.py 共用，
# 避免 st.navigation 多页面架构下的循环 import 问题。
# ============================================================


# ============================================================
# 工具函数
# ============================================================

@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    """
    全应用唯一的数据库连接入口（@st.cache_resource 保证整个 session
    乃至跨 rerun 只创建一次连接，不重复打开文件）。

    根因修复：之前这里只是 get_connection()，即 sqlite3.connect(db_path)——
    该调用在数据库文件不存在时会静默创建一个 0 张表的空文件，不会报错，
    也不会自动建表。本地开发因为开发者手动运行过 database/init_db.py，
    问题被掩盖；但 Streamlit Cloud 的每次全新部署都是干净容器，从未有人
    在其中手动跑过初始化脚本，导致 Dashboard 首次查询 matches 表时
    抛出 sqlite3.OperationalError: no such table: matches。

    现在改为返回前先调用 ensure_database_schema(conn)：该函数不依赖
    "文件是否存在"，而是直接检查"所有期望的表和列是否真的存在"，
    缺失则自动创建/重建。因为本函数是整个应用获取数据库连接的唯一入口
    （Dashboard、Match Detail、Review Center、自动导入比赛结果等所有
    代码路径都经过这里），只需要在这一处注入校验，全部调用方自动受益，
    不需要在每个使用数据库的函数里重复加校验逻辑。
    """
    conn = get_connection()
    conn = ensure_database_schema(conn)
    return conn


@st.cache_data(ttl=5)
def load_coefficients():
    """读取模型系数与标准化参数（由 worldcup_predictor_v3.py 训练时导出）。"""
    if not os.path.exists(COEF_PATH):
        return None
    with open(COEF_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_feature_attribution(feat_row: sqlite3.Row, coef_data: dict, target: str = "home_goals"):
    """
    特征归因：contribution = standardized_feature_value × coefficient

    standardized_feature_value = (raw_value - scaler_mean) / scaler_scale

    标准化是必须的，因为 Ridge 系数本身是在标准化特征空间训练得到的；
    若直接用原始特征值（如 elo_diff 量级 ±400）乘系数，会让量级大的特征
    显得贡献度异常夸大，量级小的特征（如 home_adv 取值 0/1）被严重低估。
    这一步只是把训练时已经存在的 scaler 参数应用回特征值，不引入任何新计算逻辑。

    任务3修复：home_adv 是二元类别特征（1=主场，0=中立场），不是连续数值特征。
    标准化公式 (raw-mean)/scale 对它在数学上依然成立，但业务含义会失真——
    训练集里大多数比赛是主场赛事，mean≈0.5-0.7（不是0.5正中间），
    导致 home_adv=0（中立场）标准化后变成一个较大的负值，乘以系数后
    呈现出"中立场对主队不利"这种误导性的强烈负贡献。但中立场地的实际
    业务含义是"主客双方都不享有主场优势"，对结果的影响应当是 0，
    不应该被解读成对主队的惩罚。因此对 home_adv 单独处理：
      - home_adv == 0（中立场）：contribution 强制为 0，不参与标准化计算
      - home_adv == 1（主场）：正常走标准化×系数的计算路径
    无论 target 是 home_goals 还是 away_goals，这个特殊处理逻辑都一致生效
    （因为判断条件只依赖 raw_value 本身，不依赖 target 分支选了哪组系数）。
    不影响其他任何特征的计算路径，也不改变模型预测本身（只影响展示层归因）。

    返回：[(feature_name, contribution), ...] 按贡献度绝对值不排序，原始顺序返回。
    """
    if coef_data is None or feat_row is None:
        return []

    feature_cols = coef_data["feature_cols"]  # 训练时的列名（与 prediction_features 表字段名略有差异）
    # prediction_features 表字段名 -> 训练特征列名 的映射
    # （value_ratio 对应训练时的 log_value_ratio，weighted_xg/xga 同名）
    db_field_map = {
        "elo_diff": "elo_diff",
        "log_value_ratio": "value_ratio",
        "weighted_xg_diff": "weighted_xg_diff",
        "weighted_xga_diff": "weighted_xga_diff",
        "home_adv": "home_adv",
    }
    # 注：weighted_res_diff（xG 残差特征）未持久化到 prediction_features 表，
    # Phase 1 归因仅覆盖已落库的 5 个特征。

    if target == "home_goals":
        coefs = coef_data["home_goals_coef"]
        means = coef_data["scaler_home_mean"]
        scales = coef_data["scaler_home_scale"]
    else:
        coefs = coef_data["away_goals_coef"]
        means = coef_data["scaler_away_mean"]
        scales = coef_data["scaler_away_scale"]

    contributions = []
    for i, train_col in enumerate(feature_cols):
        # 找到该训练列名对应的数据库字段
        db_col = None
        for k, v in db_field_map.items():
            if k == train_col:
                db_col = v
                break
        if db_col is None or db_col not in feat_row.keys():
            continue  # 未持久化的特征（如 weighted_res_diff）跳过

        raw_value = feat_row[db_col]
        if raw_value is None:
            continue

        display_name = FEATURE_DISPLAY_NAMES.get(train_col, train_col)

        # 任务3：home_adv 在中立场地（raw_value == 0）时，贡献度直接归零，
        # 跳过标准化×系数的通用计算路径，避免出现误导性的负贡献展示。
        if train_col == "home_adv" and raw_value == 0:
            contributions.append((display_name, 0.0))
            continue

        mean = means[i]
        scale = scales[i] if scales[i] != 0 else 1.0
        standardized = (raw_value - mean) / scale
        contribution = standardized * coefs[i]

        contributions.append((display_name, contribution))

    return contributions


def format_contribution_pct(contributions, top_pos=3, top_neg=2):
    """
    把贡献度列表格式化成 Top N 正向 / Top N 负向的展示字符串。
    贡献度本身是对 λ（进球数）的影响量，这里转成相对百分比展示，
    用 ✓ 表示正向、⚠ 表示负向。
    """
    if not contributions:
        return [], []

    total_abs = sum(abs(c) for _, c in contributions) or 1.0

    pos = sorted([c for c in contributions if c[1] > 0], key=lambda x: -x[1])[:top_pos]
    neg = sorted([c for c in contributions if c[1] < 0], key=lambda x: x[1])[:top_neg]

    pos_strs = [f"✓ {name} +{abs(val)/total_abs*100:.0f}%" for name, val in pos]
    neg_strs = [f"⚠ {name} -{abs(val)/total_abs*100:.0f}%" for name, val in neg]

    return pos_strs, neg_strs


def needs_review(prob_home, prob_draw, prob_away, has_decision) -> bool:
    """
    判断是否"需要复查"：模型给出高置信度观点（任一结果概率 >= 55%），
    但用户尚未做出决策。
    """
    max_prob = max(prob_home or 0, prob_draw or 0, prob_away or 0)
    return (max_prob >= 0.55) and (not has_decision)


# ============================================================
# Step 3 新增：博彩赔率读取（schedule_2026_result.xls）
# ============================================================

def american_to_prob(odds: float) -> float:
    """
    美式赔率 -> 隐含概率（单一结果，未归一化）。
    与 model_evaluation.py 中的同名函数逻辑完全一致，保持系统内换算口径统一。
    示例：-238 -> 0.704, +350 -> 0.222, +750 -> 0.118
    """
    if odds < 0:
        return -odds / (-odds + 100)
    else:
        return 100 / (odds + 100)


def normalize_three(p_home: float, p_draw: float, p_away: float):
    """将三个概率归一化使其和为 1（去除博彩抽水）。"""
    total = p_home + p_draw + p_away
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / total, p_draw / total, p_away / total


@st.cache_data(ttl=60)
def load_bookmaker_odds():
    """
    读取 schedule_2026_result.xls 的全部赔率行，按 match_id 建立索引。

    注意：与 model_evaluation.py 的 load_worldcup_result_data 不同，
    这里不要求 result/home_goal/away_goal/xG 字段非空——未开赛的比赛
    同样需要在 Match Detail 的"博彩公司"栏展示赔率，只有赔率三列
    （odd_win/odd_draw/odd_lose）缺失才视为该场无博彩数据。

    返回：{match_id(str): {"odd_win":..., "odd_draw":..., "odd_lose":...,
                          "result":..., "home_goal":..., "away_goal":...,
                          "home_xg":..., "away_xg":...}}
         文件不存在或读取失败时返回空字典（调用方据此显示"暂无博彩数据"）。
    """
    if not os.path.exists(WORLDCUP_RESULT_PATH):
        return {}

    try:
        df = pd.read_excel(WORLDCUP_RESULT_PATH)
        df.columns = df.columns.str.strip()
    except Exception:
        return {}

    odds_required = ["odd_win", "odd_draw", "odd_lose"]
    if not all(c in df.columns for c in odds_required):
        return {}

    df = df.dropna(subset=odds_required).copy()

    odds_by_match = {}
    for _, row in df.iterrows():
        mid = str(row["match_id"])
        odds_by_match[mid] = {
            "odd_win": row["odd_win"],
            "odd_draw": row["odd_draw"],
            "odd_lose": row["odd_lose"],
            "result": row.get("result"),
            "home_goal": row.get("home_goal"),
            "away_goal": row.get("away_goal"),
            "home_xg": row.get("home_xg"),
            "away_xg": row.get("away_xg"),
        }
    return odds_by_match


def get_bookmaker_probs_for_match(match_id) -> dict:
    """
    取某场比赛的博彩隐含概率（已归一化）。
    返回 {"p_home":..., "p_draw":..., "p_away":..., "available": bool}
    available=False 表示该场无博彩数据（赔率缺失或文件不存在）。
    """
    odds_by_match = load_bookmaker_odds()
    row = odds_by_match.get(str(match_id))
    if row is None:
        return {"p_home": None, "p_draw": None, "p_away": None, "available": False}

    raw_home = american_to_prob(row["odd_win"])
    raw_draw = american_to_prob(row["odd_draw"])
    raw_away = american_to_prob(row["odd_lose"])
    p_home, p_draw, p_away = normalize_three(raw_home, raw_draw, raw_away)
    return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away, "available": True}


# ============================================================
# Step 3 新增：伤病球员姓名读取（Injury_report.xlsx）
# ============================================================

@st.cache_data(ttl=60)
def load_injury_report():
    """
    读取 Injury_report.xlsx 全部记录。
    返回 DataFrame，列至少包含 team, player, date；文件不存在时返回空 DataFrame。
    """
    if not os.path.exists(INJURY_REPORT_PATH):
        return pd.DataFrame(columns=["team", "player", "value", "impact_score", "date"])

    try:
        df = pd.read_excel(INJURY_REPORT_PATH)
        df.columns = df.columns.str.strip()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame(columns=["team", "player", "value", "impact_score", "date"])


def get_injured_players(team: str, match_date) -> list:
    """
    取某队在 match_date 当天或之前报告的伤病球员姓名列表（不含 impact_score，
    产品要求 Match Detail 只展示姓名，不展示内部评分数值）。
    """
    df = load_injury_report()
    if df.empty:
        return []

    try:
        match_date = pd.to_datetime(match_date)
    except Exception:
        return []

    mask = (df["team"] == team) & (df["date"] == match_date)
    players = df.loc[mask, "player"].dropna().unique().tolist()
    return players


# ============================================================
# 用户决策状态聚合（Google Sheets + SQLite results 对齐）
# is_choice_correct / build_user_decision_map 已迁移至 shared_utils.py
# ============================================================


def compute_user_accuracy(decision_map: dict, results_by_match: dict):
    """
    计算"已结束比赛中用户判断正确的比例"。

    decision_map: {match_id(str): record}，来自 build_user_decision_map
    results_by_match: {match_id(str): sqlite3.Row(results表行)}，
                      只包含 is_finished=True 的比赛

    返回：(accuracy: float或None, n_correct: int, n_finished_decided: int)
         若没有任何"已决策且已结束"的比赛，accuracy 为 None（避免除以0产生误导性的0%）。
    """
    n_correct = 0
    n_total = 0
    for mid, record in decision_map.items():
        result_row = results_by_match.get(mid)
        if result_row is None:
            continue  # 比赛未结束或无结果数据，不计入准确率分母
        n_total += 1
        if is_choice_correct(record.get("choice"), result_row["result"]):
            n_correct += 1

    if n_total == 0:
        return None, 0, 0
    return n_correct / n_total, n_correct, n_total


def build_recent_timeline(decision_map: dict, matches: list, results_by_match: dict, n: int = 5):
    """
    构造"最近 n 场已结束且用户已决策的比赛"时间线，按比赛日期降序
    （最近的比赛在前），每项标记 ✅/❌。

    matches: get_all_matches_with_latest_prediction 返回的全部比赛列表，
             用于获取 home_team/away_team/date 等展示信息。
    """
    match_info_by_id = {str(m["match_id"]): m for m in matches}

    timeline_items = []
    for mid, record in decision_map.items():
        result_row = results_by_match.get(mid)
        if result_row is None:
            continue  # 未结束的比赛不进入复盘时间线
        m = match_info_by_id.get(mid)
        if m is None:
            continue

        correct = is_choice_correct(record.get("choice"), result_row["result"])
        timeline_items.append({
            "match_id": mid,
            "date": m["date"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "choice": record.get("choice"),
            "home_goals": result_row["home_goals"],
            "away_goals": result_row["away_goals"],
            "correct": correct,
        })

    timeline_items.sort(key=lambda x: x["date"], reverse=True)
    return timeline_items[:n]


# ============================================================
# Dashboard 视图
# ============================================================

def render_dashboard(conn: sqlite3.Connection, user_name: str):
    # ---- Welcome 横幅 ----
    st.title("决策中心 / Dashboard")
    st.subheader(f"Welcome back, {user_name} 👋")
    st.caption("今天我需要做哪些决策？")

    matches = get_all_matches_with_latest_prediction(conn)

    if not matches:
        st.info("暂无比赛数据。请先运行 worldcup_predictor_v4.py 生成预测。")
        return

    # ---- 从 Google Sheets 读取该用户全部决策记录（session 内缓存，任务7）----
    user_history = get_cached_user_history(user_name)
    decision_map = build_user_decision_map(user_history)

    # ---- 从 SQLite 取已结束比赛的结果（仅用于判断对错，决策本身不依赖 SQLite）----
    results_by_match = {}
    for m in matches:
        ctx = get_match_full_context(conn, m["match_id"])
        result_row = ctx["result"]
        if result_row is not None and result_row["is_finished"]:
            results_by_match[str(m["match_id"])] = result_row

    # ---- 准确率 ----
    accuracy, n_correct, n_decided_finished = compute_user_accuracy(decision_map, results_by_match)

    # ---- 已决策 / 待决策 / 需要复查 统计 ----
    total = len(matches)
    decided_count = 0
    review_count = 0
    has_decision_flags = {}

    for m in matches:
        mid = str(m["match_id"])
        has_decision = mid in decision_map
        has_decision_flags[mid] = has_decision
        if has_decision:
            decided_count += 1
        if needs_review(m["prob_home"], m["prob_draw"], m["prob_away"], has_decision):
            review_count += 1

    # ---- 顶部统计：4 个指标（新增准确率）----
    col0, col1, col2, col3 = st.columns(4)
    if accuracy is not None:
        col0.metric("决策准确率", f"{accuracy*100:.0f}%", help=f"{n_correct}/{n_decided_finished} 场（仅统计已结束且已决策的比赛）")
    else:
        col0.metric("决策准确率", "暂无数据", help="需要至少一场已结束且你已做出决策的比赛")
    col1.metric("待决策比赛", f"{total - decided_count} / {total}")
    col2.metric("已决策", decided_count)
    col3.metric("需要复查", review_count, delta=None,
                delta_color="inverse" if review_count > 0 else "normal")

    # ---- 最近5场决策结果时间线 ----
    timeline = build_recent_timeline(decision_map, matches, results_by_match, n=5)
    if timeline:
        st.write("**最近决策复盘**")
        timeline_cols = st.columns(len(timeline))
        for col, item in zip(timeline_cols, timeline):
            with col:
                icon = "✅" if item["correct"] else "❌"
                st.write(f"{icon} {item['home_team']} {item['home_goals']}-{item['away_goals']} {item['away_team']}")
                st.caption(f"你选：{item['choice']}")

    st.divider()

    # ---- 比赛卡片列表：按 match_id 降序（最新的在前）----
    matches_sorted = sorted(matches, key=lambda m: m["match_id"], reverse=True)

    for m in matches_sorted:
        mid = str(m["match_id"])
        record = decision_map.get(mid)
        has_decision = record is not None
        review_flag = needs_review(m["prob_home"], m["prob_draw"], m["prob_away"], has_decision)

        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 3, 2])

            with c1:
                st.subheader(f"{m['home_team']} vs {m['away_team']}")
                st.caption(f"{m['date']} · {m['venue'] or '场地待定'}")

            with c2:
                if m["prob_home"] is not None:
                    st.write("**模型观点**")
                    st.write(
                        f"主胜 {m['prob_home']*100:.0f}% · "
                        f"平局 {m['prob_draw']*100:.0f}% · "
                        f"客胜 {m['prob_away']*100:.0f}%"
                    )
                    # 任务2：预测比分，直接复用模型 λ 计算结果，不重新建模
                    if m["lambda_home"] is not None and m["lambda_away"] is not None:
                        pred_h, pred_a, pred_p = most_likely_score(m["lambda_home"], m["lambda_away"])
                        st.caption(f"预测比分：{pred_h} : {pred_a}（{pred_p*100:.1f}%）")
                else:
                    st.write("模型观点：暂无预测")

            with c3:
                # 用户决策状态比模型预测更醒目：放在最右、用色块强调
                if has_decision:
                    st.success(f"✅ 已决策：{record.get('choice')}")
                    if record.get("confidence"):
                        st.caption(f"信心：{record.get('confidence')}/5")
                else:
                    if review_flag:
                        st.error("🔴 需要复查：模型高置信度，你尚未决策")
                    else:
                        st.warning("⏳ 待决策")

            btn_label = "查看详情 / 修改决策" if has_decision else "做出决策"
            if st.button(btn_label, key=f"goto_{m['match_id']}"):
                st.session_state["selected_match_id"] = m["match_id"]
                st.session_state["view"] = "detail"
                st.rerun()


# ============================================================
# Dixon-Coles 比分计算（most_likely_score）已迁移至 shared_utils.py
# ============================================================


# ============================================================
# Match Detail 视图（Step 3：三栏对比，完全迁移到 Google Sheets）
# ============================================================

def render_match_detail(conn: sqlite3.Connection, match_id, user_name: str):
    ctx = get_match_full_context(conn, match_id)
    match = ctx["match"]
    pred = ctx["prediction"]
    feat = ctx["features"]
    result = ctx["result"]

    if match is None:
        st.error("未找到该比赛数据。")
        if st.button("返回 Dashboard"):
            st.session_state["view"] = "dashboard"
            st.rerun()
        return

    if st.button("← 返回 Dashboard"):
        st.session_state["view"] = "dashboard"
        st.rerun()

    st.title(f"{match['home_team']} vs {match['away_team']}")

    # ---- 任务3：已结束比赛在标题后显示真实比分 + 预期进球（xG）----
    # 数据来源：schedule_2026_result.xls 的 home_goal/away_goal/home_xg/away_xg。
    # 是否"已结束"仍以 SQLite results 表的 is_finished 为准（保持现有数据源选择不变），
    # 比分/xG 数值本身从 Excel 文件读取（results 表未存储 xG）。
    odds_info = get_bookmaker_probs_for_match(match_id)
    odds_raw = load_bookmaker_odds().get(str(match_id), {})

    if result is not None and result["is_finished"]:
        st.write(f"**比赛结果：{result['home_goals']} : {result['away_goals']}**")

        actual_home_xg = odds_raw.get("home_xg")
        actual_away_xg = odds_raw.get("away_xg")
        # Bug修复（任务2）：pandas 读取 Excel 时缺失的数值会变成 NaN（float类型），
        # 不是 None。`NaN is not None` 恒为 True，原判断会让 NaN 漏网，
        # 格式化后显示成字面文本 "nan : nan"。改用 pd.notna() 同时排除
        # None 和 NaN 两种缺失值表示形式。
        if pd.notna(actual_home_xg) and pd.notna(actual_away_xg):
            st.write(f"**预期进球：{actual_home_xg:.1f} : {actual_away_xg:.1f}**")

        st.caption(f"{match['date']} · {match['venue'] or '场地待定'}")
    else:
        # 比赛尚未结束：保持原有展示方式（不显示比分/xG，仅显示时间地点）
        st.caption(f"{match['date']} · {match['venue'] or '场地待定'}")

    st.divider()

    # ---- 取该用户在 Google Sheets 中对本场比赛的最新决策（session 内缓存，任务7）----
    user_history = get_cached_user_history(user_name)
    match_records = [r for r in user_history if str(r.get("match_id", "")) == str(match_id)]
    match_records.sort(key=lambda r: r.get("timestamp", ""))
    latest = match_records[-1] if match_records else None

    # 任务4：理由下拉标签 + 自由文本，合并存入同一个 reason 字段（不改数据库结构）。
    # 存储格式："标签｜自由文本"，读取时按"｜"反解析回两个控件的初始值。
    latest_reason_tag, latest_reason_text = "其他", ""
    if latest and latest.get("reason"):
        raw_reason = latest.get("reason", "")
        if "｜" in raw_reason:
            tag_part, _, text_part = raw_reason.partition("｜")
            if tag_part in REASON_TAG_OPTIONS:
                latest_reason_tag = tag_part
                latest_reason_text = text_part
            else:
                latest_reason_text = raw_reason
        else:
            latest_reason_text = raw_reason

    # AI 当前判断（用于：(a) 写决策时存快照，(b) 中栏展示）
    # Step 5：复用 shared_utils.get_ai_choice_and_confidence，
    # 与 Review Center 展示 AI 置信度时使用同一口径，避免重复实现导致漂移。
    if pred is not None:
        ai_choice, ai_confidence = get_ai_choice_and_confidence(
            pred["prob_home"], pred["prob_draw"], pred["prob_away"]
        )
    else:
        ai_choice, ai_confidence = None, None

    # ============================================================
    # 三栏对比：左=用户决策，中=AI预测，右=博彩公司预测
    # ============================================================
    col_user, col_ai, col_book = st.columns(3)

    # ---- 左栏：用户预测 ----
    with col_user:
        st.subheader("你的预测")

        if latest:
            st.success(f"当前决策：**{latest.get('choice')}**　|　信心：{latest.get('confidence')}/5")
            if latest.get("reason"):
                st.caption(f"理由：{latest.get('reason')}")

        with st.form(key=f"decision_form_{match_id}"):
            choice = st.radio(
                "你认为结果是？",
                options=["home", "draw", "away"],
                format_func=lambda x: {"home": f"{match['home_team']} 胜",
                                       "draw": "平局",
                                       "away": f"{match['away_team']} 胜"}[x],
                index=["home", "draw", "away"].index(latest["choice"])
                if latest and latest.get("choice") in ["home", "draw", "away"] else 0,
            )
            confidence = st.slider("你的信心程度", 1, 5,
                                   value=int(latest["confidence"]) if latest and latest.get("confidence") else 3)

            # 任务4：恢复决策理由下拉菜单，同时保留"你的看法"自由文本输入框
            reason_tag = st.selectbox(
                "决策理由",
                options=REASON_TAG_OPTIONS,
                index=REASON_TAG_OPTIONS.index(latest_reason_tag)
                if latest_reason_tag in REASON_TAG_OPTIONS else 0,
            )
            reason_text = st.text_area("你的看法（可选）", value=latest_reason_text)

            submitted = st.form_submit_button("保存决策")

            if submitted:
                # 两个输入项都需要保存：合并为单一 reason 字段写入（不改数据库结构）
                combined_reason = f"{reason_tag}｜{reason_text}" if reason_text else reason_tag

                ok = save_decision(
                    user_name=user_name,
                    match_id=match_id,
                    choice=choice,
                    confidence=confidence,
                    reason=combined_reason,
                    ai_choice=ai_choice,
                    ai_confidence=ai_confidence,
                )
                if ok:
                    invalidate_user_history_cache()  # 任务7：写入成功后让缓存失效
                    st.success("决策已保存！")
                else:
                    st.warning("决策未能持久化（Google Sheets 不可用），但不影响继续使用。")
                st.rerun()

        if match_records:
            with st.expander(f"决策历史（共 {len(match_records)} 次）"):
                for h in match_records:
                    st.write(f"- {h.get('timestamp')} · 选择：{h.get('choice')} · "
                            f"信心：{h.get('confidence')}"
                            + (f" · 理由：{h.get('reason')}" if h.get("reason") else ""))

    # ---- 中栏：AI 预测 ----
    with col_ai:
        st.subheader("AI 预测")

        if pred is not None:
            st.write(f"主胜 {pred['prob_home']*100:.1f}% · "
                    f"平局 {pred['prob_draw']*100:.1f}% · "
                    f"客胜 {pred['prob_away']*100:.1f}%")
            st.write(f"**λ（预期进球）**：{pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}")

            best_h, best_a, best_p = most_likely_score(pred["lambda_home"], pred["lambda_away"])
            st.write(f"**最可能比分**：{best_h}-{best_a} ({best_p*100:.1f}%)")

            st.caption(f"模型版本：{pred['model_version']}")
        else:
            st.info("暂无模型预测数据。")

    # ---- 右栏：博彩公司预测 ----
    with col_book:
        st.subheader("博彩公司预测")

        if odds_info["available"]:
            st.write(f"主胜 {odds_info['p_home']*100:.1f}% · "
                    f"平局 {odds_info['p_draw']*100:.1f}% · "
                    f"客胜 {odds_info['p_away']*100:.1f}%")
            st.caption(f"美式赔率：{odds_raw.get('odd_win')} / "
                      f"{odds_raw.get('odd_draw')} / {odds_raw.get('odd_lose')}")
            st.caption("赔率已去除博彩抽水（归一化处理）")
        else:
            st.info("暂无博彩赔率数据。")

    st.divider()

    # ============================================================
    # 球队信息（含伤病球员姓名，不显示 impact_score）
    # ============================================================
    st.header("球队信息")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(match["home_team"])
        st.write(f"Elo：{match['home_elo']:.0f}")
        st.write(f"身价：€{match['home_value']:.1f}M")
        injured_home = get_injured_players(match["home_team"], match["date"])
        if injured_home:
            st.write(f"伤病球员：{', '.join(injured_home)}")
        else:
            st.caption("无伤病报告")
    with c2:
        st.subheader(match["away_team"])
        st.write(f"Elo：{match['away_elo']:.0f}")
        st.write(f"身价：€{match['away_value']:.1f}M")
        injured_away = get_injured_players(match["away_team"], match["date"])
        if injured_away:
            st.write(f"伤病球员：{', '.join(injured_away)}")
        else:
            st.caption("无伤病报告")

    st.divider()

    # ============================================================
    # 特征解释（简化：只展示 Top Positive 和 Top Negative 各一项）
    # ============================================================
    st.header("特征解释")
    st.caption("contribution = standardized_feature_value × coefficient（非 LLM，直接来自 Ridge 系数）")

    coef_data = load_coefficients()
    if coef_data is None:
        st.warning("未找到 model_coefficients.json，请先运行 worldcup_predictor_v4.py。")
    elif feat is None:
        st.warning("该比赛暂无特征记录。")
    else:
        contributions = compute_feature_attribution(feat, coef_data, target="home_goals")
        pos_strs, neg_strs = format_contribution_pct(contributions, top_pos=1, top_neg=1)

        if pos_strs or neg_strs:
            line = "　".join(pos_strs + neg_strs)
            st.write(line)
        else:
            st.info("暂无可归因的特征贡献。")


# ============================================================
# 主入口
# ============================================================

def run_dashboard_page():
    """
    st.navigation 的 Dashboard 页面入口。
    内部仍保留 Step 1-4 已有的 session_state["view"] 机制，
    用于在同一页面内切换 Dashboard 列表视图和 Match Detail 详情视图——
    这是 app.py 内部视图切换，与 st.navigation 的多页面路由是两个不同层级，
    互不冲突（st.navigation 负责"Dashboard页面" vs "Review Center页面"之间跳转）。
    """
    user_name = get_current_user()
    if not user_name:
        render_login_gate()
        return

    if "view" not in st.session_state:
        st.session_state["view"] = "dashboard"
    if "selected_match_id" not in st.session_state:
        st.session_state["selected_match_id"] = None

    conn = get_db_connection()

    with st.sidebar:
        st.caption(f"当前用户：{user_name}")
        if st.button("🏠 回到 Dashboard 列表"):
            st.session_state["view"] = "dashboard"
            st.rerun()

    if st.session_state["view"] == "dashboard" or st.session_state["selected_match_id"] is None:
        render_dashboard(conn, user_name)
    else:
        render_match_detail(conn, st.session_state["selected_match_id"], user_name)


# ============================================================
# 任务2：应用启动时自动导入比赛结果（幂等，避免重复导入）
# ============================================================

def _results_table_has_data(conn: sqlite3.Connection) -> bool:
    """
    检查 SQLite 的 results 表是否已经存在至少一条 is_finished=1 的记录。
    用作自动导入的幂等性判断依据：数据库里有数据就跳过，没有才导入。

    注意：这里用数据库状态本身做判断，而不是 st.session_state——
    因为 session_state 只在单个浏览器 session 内有效，Streamlit Cloud
    上不同用户访问会各自建立新 session，如果用 session_state 判断，
    每个新用户首次访问都会重新触发一次导入（虽然 save_result 是幂等的，
    不会产生脏数据，但会造成不必要的重复 IO）。用数据库状态判断则是
    真正"全局只导入一次"，与谁访问、访问几次无关。
    """
    cur = conn.execute("SELECT COUNT(*) FROM results WHERE is_finished = 1")
    count = cur.fetchone()[0]
    return count > 0


def _auto_import_results_if_needed(conn: sqlite3.Connection) -> None:
    """
    应用启动时调用：如果 results 表为空（没有任何已结束比赛记录），
    自动从 data/schedule_2026_result.xls 导入一次。

    失败时只打印警告，不阻断应用启动（与本项目其他外部数据源
    一致的降级原则：缺数据时功能降级展示，不让整个应用崩溃）。
    """
    try:
        if _results_table_has_data(conn):
            return  # 已有数据，跳过导入

        from import_results import import_results
        imported = import_results()
        if imported > 0:
            print(f"[app.py] 启动时自动导入了 {imported} 场已结束比赛结果。")
    except Exception as e:
        print(f"[app.py] 警告：启动时自动导入比赛结果失败（{e}），"
              "Match Detail 页面的真实比分/xG 展示可能暂时不可用。")


def main():
    st.set_page_config(page_title="Decision Review Copilot", layout="wide")

    # ---- 任务2：启动时自动导入比赛结果（幂等）----
    _auto_import_results_if_needed(get_db_connection())

    # ---- st.navigation 多页面入口 ----
    # Dashboard（本文件）与 Review Center（pages/review.py）通过 st.navigation
    # 统一注册，取代旧版 Streamlit 仅靠 pages/ 目录文件名自动发现导航项的方式，
    # 可以自定义标题、图标，且导航结构在代码中显式可见。
    dashboard_page = st.Page(
        run_dashboard_page,
        title="Dashboard",
        icon="🏠",
        default=True,
    )
    review_page = st.Page(
        "review.py",
        title="Review Center",
        icon="📋",
    )

    nav = st.navigation([dashboard_page, review_page])
    nav.run()


if __name__ == "__main__":
    main()
