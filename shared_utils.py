"""
shared_utils.py
================
跨页面共享的工具函数（用户身份、决策判定、Dixon-Coles 比分计算）。

为什么需要这个文件：
  在引入 st.navigation 之前，pages/review.py 曾直接 `from app import ...`
  复用 app.py 里的函数。但 st.Page("pages/review.py", ...) 这种文件路径调度
  方式下，Streamlit 会把目标文件当独立脚本执行，"反向 import 主入口文件"
  容易引发模块重复加载、session_state 隔离异常等问题。

  因此把两边都需要的纯函数（不依赖 app.py 的页面渲染逻辑）抽到本文件，
  app.py 和 pages/review.py 都从这里导入，不再互相依赖。

本文件不包含任何 st.* 渲染调用（除 get_current_user/set_current_user/
render_login_gate 必须用到的 st.query_params 和基础表单控件），
保持纯函数风格，方便测试和复用。
"""

import streamlit as st
import pandas as pd


# ============================================================
# 用户身份
# ============================================================
#
# UI修复任务1：页面切换时身份丢失的根因——
# st.navigation 在浏览器中切换 st.Page 时，前端路由是否保留 URL
# query string 不是 Python 后端能直接控制的行为（这是 Streamlit
# 多页应用一个已知的边界情况）。因此不能只依赖 st.query_params
# 作为身份的唯一来源。
#
# 修复方式：以 st.session_state 作为身份的权威来源（同一浏览器
# session 内，session_state 在 st.navigation 的所有页面之间是
#持续存在的，这是 Streamlit 官方保证的行为，不受前端路由影响）。
# st.query_params 仍然维护，用于支持"复制链接分享"等场景，
# 但读取身份时优先信任 session_state，如果 session_state 没有
# 但 query_params 有（例如用户直接用带参数的链接打开），则用
# query_params 的值回填 session_state。
#
# 登录方式本身没有变化：依然是首次访问无身份时显示昵称输入框，
# 提交后即可在本次 session 内畅通切换所有页面。

_SESSION_KEY_USER = "user_name"

# UI修复任务4：决策理由下拉选项（恢复）
REASON_TAG_OPTIONS = ["伤病", "状态", "主场优势", "历史交锋", "身价差距", "直觉", "其他"]


def get_current_user():
    """
    返回当前用户昵称，找不到则返回 None（由调用方显示登录页）。

    优先级：
      1. st.session_state["user_name"] —— 权威来源，跨 st.Page 切换不丢失
      2. st.query_params["user"] —— 仅在 session_state 还没有时使用
         （例如用户首次用带 ?user= 的链接直接打开某个子页面），
         读取到后立刻同步回 session_state，确保后续不再依赖 query_params。
    """
    if _SESSION_KEY_USER in st.session_state and st.session_state[_SESSION_KEY_USER]:
        return st.session_state[_SESSION_KEY_USER]

    params = st.query_params
    user = params.get("user", None)
    if user:
        user = str(user).strip()
        if user:
            st.session_state[_SESSION_KEY_USER] = user
            return user

    return None


def set_current_user(user_name: str):
    """
    同时写入 session_state（权威来源，跨页面切换不丢失）和
    query_params（用于浏览器地址栏展示、支持分享链接）。
    """
    st.session_state[_SESSION_KEY_USER] = user_name
    st.query_params["user"] = user_name


def render_login_gate():
    """
    首次访问（session 内还没有身份）时展示的昵称输入页。
    提交后写入 session_state + query_params 并 rerun。
    """
    st.title("欢迎来到 Decision Review Copilot")
    st.write("这是一个帮助你记录决策、复盘判断质量的工具，不是博彩预测工具。")
    st.write("请输入你的昵称以开始：")

    with st.form(key="login_form"):
        nickname = st.text_input("你的昵称", placeholder="例如：Tom")
        submitted = st.form_submit_button("进入 Dashboard")

        if submitted:
            nickname = nickname.strip()
            if not nickname:
                st.error("昵称不能为空。")
            else:
                set_current_user(nickname)
                st.rerun()


# ============================================================
# UI修复任务7：Google Sheets 决策记录的 session 内缓存
# ============================================================
#
# 目标：Dashboard / Match Detail / Review Center 三处都需要读取
# 当前用户的决策记录，但不应该每次切换页面、每次 rerun 都重新打
# Google Sheets API。改为：同一用户的决策记录在 session 内只读取
# 一次，缓存进 session_state；只有调用 save_decision 成功写入新
# 决策后，才显式让缓存失效，下次读取时才会重新拉取。

_SESSION_KEY_HISTORY_CACHE = "user_history_cache"
_SESSION_KEY_HISTORY_CACHE_USER = "user_history_cache_user"


def get_cached_user_history(user_name: str, force_refresh: bool = False) -> list:
    """
    返回该用户的决策记录列表，session 内只真正调用一次
    database.db.get_user_history（除非 force_refresh=True 或换了用户）。

    用法：
      - Dashboard / Match Detail / Review Center 读取决策记录时一律调用
        本函数，不直接调用 database.db.get_user_history。
      - 提交决策成功后调用 invalidate_user_history_cache()，
        下一次调用本函数时会重新拉取最新数据。
    """
    from database.db import get_user_history  # 延迟导入，避免模块级循环依赖

    cached_user = st.session_state.get(_SESSION_KEY_HISTORY_CACHE_USER)
    cached_data = st.session_state.get(_SESSION_KEY_HISTORY_CACHE)

    needs_fetch = (
        force_refresh
        or cached_data is None
        or cached_user != user_name
    )

    if needs_fetch:
        cached_data = get_user_history(user_name)
        st.session_state[_SESSION_KEY_HISTORY_CACHE] = cached_data
        st.session_state[_SESSION_KEY_HISTORY_CACHE_USER] = user_name

    return cached_data


def invalidate_user_history_cache():
    """
    让决策记录缓存失效。在 save_decision 成功写入后调用，
    确保下一次 get_cached_user_history 会重新从 Google Sheets 拉取，
    使刚提交的决策立刻反映在 Dashboard / Match Detail / Review Center 上。
    """
    st.session_state[_SESSION_KEY_HISTORY_CACHE] = None


# ============================================================
# 决策正确性判定（Dashboard / Match Detail / Review Center 共用）
# ============================================================

RESULT_TO_CHOICE = {"H": "home", "D": "draw", "A": "away"}


def is_choice_correct(choice: str, result_code: str) -> bool:
    """
    比较用户选择（'home'/'draw'/'away'）与 SQLite results 表的 result 字段
    （'H'/'D'/'A'）是否一致。result_code 为 None 或未知值时返回 False。
    """
    if not choice or not result_code:
        return False
    return RESULT_TO_CHOICE.get(str(result_code).strip().upper()) == choice


def build_user_decision_map(user_history: list) -> dict:
    """
    把 Google Sheets 返回的扁平记录列表，按 match_id 聚合为
    "该场比赛该用户的最新一条决策"（同一场比赛可能有多次提交，
    取 timestamp 最大的一条作为当前决策，与 save_decision 的
    "只追加不修改"语义对应——最新追加的代表当前有效决策）。

    返回：{match_id(str): record_dict}
    """
    latest_by_match = {}
    for r in user_history:
        mid = str(r.get("match_id", ""))
        if not mid:
            continue
        existing = latest_by_match.get(mid)
        if existing is None or r.get("timestamp", "") >= existing.get("timestamp", ""):
            latest_by_match[mid] = r
    return latest_by_match


# ============================================================
# Dixon-Coles 比分计算（与 worldcup_predictor_v4.py 的 score_matrix 逐字一致）
# ============================================================

DC_RHO = -0.13
MAX_GOALS = 8


def _dc_tau(x: int, y: int, mu: float, nu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - mu * nu * rho
    elif x == 1 and y == 0:
        return 1 + nu * rho
    elif x == 0 and y == 1:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def most_likely_score(lam_home: float, lam_away: float, rho: float = DC_RHO):
    """
    从已存储的 λ 重新计算最可能比分及其概率（数据库未持久化"最可能比分"
    这个派生量，因此在展示层按需重算，不引入新的建模假设）。
    返回 (home_goals, away_goals, probability)。
    """
    from scipy.stats import poisson
    import numpy as np

    mat = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
            tau = _dc_tau(i, j, lam_home, lam_away, rho)
            mat[i, j] = p * tau
    mat /= mat.sum()

    best_i, best_j = np.unravel_index(np.argmax(mat), mat.shape)
    return int(best_i), int(best_j), float(mat[best_i, best_j])


# ============================================================
# Value Lab / 策略实验室：核心回测计算（纯函数，无 st.* 渲染调用）
# ============================================================
#
# 放在 shared_utils.py 而不是 value_lab.py 的原因：
#   value_lab.py 作为 st.Page 独立页面，文件末尾无条件调用 main()
#   渲染整个页面（与 review.py 同样的模式，见 review.py 文件末尾注释）。
#   如果 Review Center 想要复用 Value Lab 的计算逻辑，直接
#   `from value_lab import ...` 会在 import 时把 Value Lab 页面
#   整个渲染一遍（副作用），而不是只拿到计算结果。
#   因此把不含任何 st.* 渲染调用的纯计算逻辑放在 shared_utils.py，
#   value_lab.py（页面渲染）和 review.py（摘要卡片）都从这里导入，
#   两处展示的数字保证同一套口径、不重复实现。

VALUE_LAB_CHOICES = ("home", "draw", "away")

# 策略 A/B/C 共用的 EV 下注阈值
VALUE_LAB_EV_THRESHOLD = 0.05
# 策略 C 使用的模型主选置信度阈值
VALUE_LAB_CONFIDENCE_THRESHOLD = 0.5


def _to_float(v):
    """
    安全地把 Google Sheets 单元格值转为 float。
    空字符串、None、非法值一律返回 None（调用方据此判断"该字段是否缺失"）。
    """
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _build_value_lab_items(user_name: str, conn) -> list:
    """
    构建 Value Lab 回测所需的"可回测比赛列表"。

    数据来源与兼容性规则（必须严格遵守）：
      1. Google Sheet 是唯一真相来源：AI概率、市场概率、赔率、EV 全部
         使用决策时刻写入的快照字段，不重新从 SQLite 当前预测结果计算。
      2. SQLite（database.db_utils.get_match_full_context）只用于补充
         比赛名称、时间、场地和实际赛果，不参与概率/EV计算。
      3. 同一用户对同一 match_id 的多条记录，只取最新 timestamp 一条
         （复用 build_user_decision_map，与 Review Center 判定口径一致）。
      4. 缺失 market_*_prob 或原始赔率（home/draw/away_odds）→ 跳过该场。
      5. 缺失 ev_* 但 ai_*_prob 和 market_*_prob 都在 → 临时补算 EV。
      6. ai_*_prob 缺失（无论是否已有 ev_*）→ 跳过该场（策略B/C需要
         AI主选，为保证三条策略统计口径一致，缺AI概率的比赛整体不纳入）。
      7. 只统计已结束（is_finished）的比赛。

    返回：list[dict]，按比赛日期升序排列（供回测按时间顺序计算最大回撤）。
    """
    from database.db_utils import get_match_full_context  # 延迟导入，避免模块级循环依赖

    user_history = get_cached_user_history(user_name)
    decision_map = build_user_decision_map(user_history)

    items = []
    for mid, r in decision_map.items():
        market_home = _to_float(r.get("market_home_prob"))
        market_draw = _to_float(r.get("market_draw_prob"))
        market_away = _to_float(r.get("market_away_prob"))
        home_odds = _to_float(r.get("home_odds"))
        draw_odds = _to_float(r.get("draw_odds"))
        away_odds = _to_float(r.get("away_odds"))

        # 规则4：缺失市场概率或原始赔率 -> 跳过该场
        if None in (market_home, market_draw, market_away, home_odds, draw_odds, away_odds):
            continue
        if market_home <= 0 or market_draw <= 0 or market_away <= 0:
            continue

        ai_home = _to_float(r.get("ai_home_prob"))
        ai_draw = _to_float(r.get("ai_draw_prob"))
        ai_away = _to_float(r.get("ai_away_prob"))

        # 规则6：AI概率缺失 -> 跳过该场
        if None in (ai_home, ai_draw, ai_away):
            continue

        ev_home = _to_float(r.get("ev_home"))
        ev_draw = _to_float(r.get("ev_draw"))
        ev_away = _to_float(r.get("ev_away"))

        if None in (ev_home, ev_draw, ev_away):
            # 规则5：EV缺失但AI/市场概率都在 -> 临时补算（口径与写入时一致）
            try:
                ev_home = ai_home / market_home - 1
                ev_draw = ai_draw / market_draw - 1
                ev_away = ai_away / market_away - 1
            except Exception:
                continue

        # match_id 在 Google Sheet 中以字符串形式存储，但 SQLite 侧通常是
        # INTEGER 主键；尽量转成 int 再查询，转换失败（非数字ID）则按原样传入。
        try:
            lookup_id = int(mid)
        except (TypeError, ValueError):
            lookup_id = mid

        try:
            ctx = get_match_full_context(conn, lookup_id)
        except Exception:
            continue

        match = ctx.get("match")
        result = ctx.get("result")
        # 规则7：只统计已结束比赛
        if match is None or result is None or not result["is_finished"]:
            continue

        actual_choice = RESULT_TO_CHOICE.get(str(result["result"]).strip().upper())
        if actual_choice is None:
            continue

        # 归一化后的市场概率是"去水"公平概率，十进制赔率 = 1 / 市场概率，
        # 与 Review Center ROI 计算（review.py 中的 _get_decimal_odds）
        # 使用同一套"去水后概率求倒数"的口径，保证两处ROI可比。
        decimal_odds = {
            "home": 1 / market_home,
            "draw": 1 / market_draw,
            "away": 1 / market_away,
        }

        user_choice = r.get("choice") if r.get("choice") in VALUE_LAB_CHOICES else None

        items.append({
            "match_id": mid,
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "date": match["date"],
            "venue": match["venue"],
            "actual_choice": actual_choice,
            "user_choice": user_choice,
            "ai_home": ai_home, "ai_draw": ai_draw, "ai_away": ai_away,
            "market_home": market_home, "market_draw": market_draw, "market_away": market_away,
            "ev_home": ev_home, "ev_draw": ev_draw, "ev_away": ev_away,
            "decimal_odds": decimal_odds,
        })

    items.sort(key=lambda x: x["date"])
    return items


def _strategy_a_pick(item):
    """策略A：Top EV > 5%。取EV最高的结果，若其EV超过阈值则下注。"""
    ev_map = {"home": item["ev_home"], "draw": item["ev_draw"], "away": item["ev_away"]}
    best = max(ev_map, key=ev_map.get)
    if ev_map[best] > VALUE_LAB_EV_THRESHOLD:
        return best, ev_map[best]
    return None, None


def _strategy_b_pick(item):
    """策略B：Top EV + 模型主选一致。EV最高的结果必须同时是AI模型主选。"""
    ev_map = {"home": item["ev_home"], "draw": item["ev_draw"], "away": item["ev_away"]}
    ai_map = {"home": item["ai_home"], "draw": item["ai_draw"], "away": item["ai_away"]}
    model_choice = max(ai_map, key=ai_map.get)
    best = max(ev_map, key=ev_map.get)
    if ev_map[best] > VALUE_LAB_EV_THRESHOLD and best == model_choice:
        return best, ev_map[best]
    return None, None


def _strategy_c_pick(item):
    """策略C：高置信度 + 正EV。AI模型主选概率≥50%，且该结果EV超过阈值。"""
    ai_map = {"home": item["ai_home"], "draw": item["ai_draw"], "away": item["ai_away"]}
    model_choice = max(ai_map, key=ai_map.get)
    model_prob = ai_map[model_choice]
    ev_map = {"home": item["ev_home"], "draw": item["ev_draw"], "away": item["ev_away"]}
    ev_for_choice = ev_map[model_choice]
    if model_prob >= VALUE_LAB_CONFIDENCE_THRESHOLD and ev_for_choice > VALUE_LAB_EV_THRESHOLD:
        return model_choice, ev_for_choice
    return None, None


def _real_decision_pick(item):
    """"你的真人决策"：按用户在决策当时实际做出的选择计算，作为对比基准。"""
    if item["user_choice"] is None:
        return None, None
    outcome = item["user_choice"]
    ev = item.get(f"ev_{outcome}")
    return outcome, ev


STRATEGY_PICKERS = {
    "A": ("策略A：Top EV > 5%", _strategy_a_pick),
    "B": ("策略B：Top EV + 模型主选一致", _strategy_b_pick),
    "C": ("策略C：高置信度 + 正EV", _strategy_c_pick),
}


def _simulate_strategy(items: list, picker) -> dict:
    """
    对给定策略 picker 函数在 items（已按比赛日期升序排列）上做回测。

    盈亏规则：下注1单位；猜中 profit = 十进制赔率 - 1；猜错 profit = -1。
    最大回撤：按下注顺序累积 profit 形成权益曲线，取 peak-to-trough 最大值
    （单位：下注单位数）。

    返回：{
        "label": 策略名（回测结果不含label，由调用方补充）——本函数不设置该字段
        "bets": [...],           # 每场下注的明细
        "bet_count": int,
        "hit_count": int,
        "roi": float | None,     # 百分比数值，如 12.3 表示 +12.3%
        "hit_rate": float | None,# 0~1之间
        "avg_ev": float | None,
        "max_drawdown": float,
    }
    """
    bets = []
    profits = []
    for item in items:
        outcome, ev = picker(item)
        if outcome is None:
            continue
        odds = item["decimal_odds"][outcome]
        hit = (outcome == item["actual_choice"])
        profit = (odds - 1) if hit else -1.0
        bets.append({
            "match_id": item["match_id"],
            "home_team": item["home_team"],
            "away_team": item["away_team"],
            "date": item["date"],
            "outcome": outcome,
            "ev": ev,
            "odds": odds,
            "actual_choice": item["actual_choice"],
            "hit": hit,
            "profit": profit,
            # Diagnostics 模块新增：透传该下注结果对应的AI/市场概率快照
            # （数值已经存在于 item 中，这里只是多带一份，不改变任何策略
            # 判定逻辑，也不重新计算任何概率）。
            "model_prob": item.get(f"ai_{outcome}"),
            "market_prob": item.get(f"market_{outcome}"),
        })
        profits.append(profit)

    bet_count = len(bets)
    hit_count = sum(1 for b in bets if b["hit"])
    total_profit = sum(profits)

    roi = (total_profit / bet_count * 100) if bet_count > 0 else None
    hit_rate = (hit_count / bet_count) if bet_count > 0 else None
    avg_ev = (sum(b["ev"] for b in bets) / bet_count) if bet_count > 0 else None

    max_dd = 0.0
    cum = 0.0
    peak = 0.0
    for p in profits:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    return {
        "bets": bets,
        "bet_count": bet_count,
        "hit_count": hit_count,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_ev": avg_ev,
        "max_drawdown": max_dd,
    }


def compute_value_lab_results(user_name: str, conn) -> dict:
    """
    Value Lab 页面与 Review Center「Value Bet Snapshot」摘要卡片共用的
    核心回测入口（纯函数，不含任何 st.* 渲染调用）。

    返回：{
        "eligible_count": int,               # 有完整快照、已结束的比赛数
        "strategies": {"A": {...}, "B": {...}, "C": {...}},  # 见 _simulate_strategy
        "real": {...},                       # 用户真实决策的同口径回测结果
    }
    每个策略结果 dict 额外含 "label" 字段（策略中文名称）。
    """
    items = _build_value_lab_items(user_name, conn)

    strategies = {}
    for key, (label, picker) in STRATEGY_PICKERS.items():
        result = _simulate_strategy(items, picker)
        result["label"] = label
        strategies[key] = result

    real_result = _simulate_strategy(items, _real_decision_pick)
    real_result["label"] = "你的真人决策"

    return {
        "eligible_count": len(items),
        "strategies": strategies,
        "real": real_result,
    }


def get_ai_choice_and_confidence(prob_home: float, prob_draw: float, prob_away: float):
    """
    Step 5 新增：从 AI 的 WDL 概率中取出"AI 的选择"和"AI 对该选择的置信度"。
    AI 选择 = 概率最大的那个结果；置信度 = 该结果对应的概率值。

    这是 save_decision 写入 ai_choice/ai_confidence 字段时使用的唯一口径，
    Match Detail、Review Center 展示 AI 置信度时也复用同一个函数，
    保证"AI 当时怎么想"在写入快照和事后复盘展示之间不会出现口径不一致。

    返回：(ai_choice: str, ai_confidence: float)
    """
    probs = {"home": prob_home, "draw": prob_draw, "away": prob_away}
    ai_choice = max(probs, key=probs.get)
    ai_confidence = probs[ai_choice]
    return ai_choice, ai_confidence


# ============================================================
# Value Lab / Diagnostics：策略A（Top EV>5%）系统性偏差分析
# ============================================================
#
# 设计原则（与 Value Lab 页面 review.py 顶部注释一致的"唯一真相来源"精神）：
#   1. 只消费 compute_value_lab_results() 已经算好的策略A下注明细，
#      不重新计算AI概率，不修改策略A/B/C的定义与回测逻辑。
#   2. 所有4张诊断表都从同一张 strategy_a_diagnostics_df 里 groupby 生成，
#      避免每张表各写一套聚合逻辑。
#   3. Elo 数据缺失（SQLite 里查不到、或该批比赛本身没有 Elo）时，
#      只让"按Elo差距分层"这一张表整体降级为不可用，不影响其余3张表，
#      也不抛异常中断整个 Diagnostics 区块。

# 诊断表2 使用的赔率分层边界（左闭右开，低于最小档/高于最大档的极端值
# 分别归入首档/末档，不会产生未分类的行）。
ODDS_BUCKET_ORDER = ["1.30-1.80", "1.80-2.50", "2.50-4.00", "4.00+"]

# 诊断表3 使用的"模型-市场分歧"分层边界（prob_gap 为 0~1 之间的小数）。
GAP_BUCKET_ORDER = ["0-5%", "5-10%", "10-15%", "15%+"]

# 诊断表4 使用的 Elo 差距分层边界。
ELO_BUCKET_ORDER = ["0-80", "80-180", "180-300", "300+"]

# 单个分组下注场次少于该值时，即使命中了结论规则，也在 UI 上额外提示
# "样本量过小，结论仅供参考"，避免小样本被误读为系统性偏差。
DIAGNOSTICS_MIN_SAMPLE_SIZE = 5

BET_TYPE_LABEL = {"home": "主胜", "draw": "平局", "away": "客胜"}
BET_TYPE_ORDER = ["home", "draw", "away"]


def _bucket_odds(odds):
    """赔率分层：< 1.80 归入首档（含所有低于1.30的极端值），>= 4.00 归入末档。"""
    if odds is None:
        return None
    if odds < 1.80:
        return "1.30-1.80"
    elif odds < 2.50:
        return "1.80-2.50"
    elif odds < 4.00:
        return "2.50-4.00"
    else:
        return "4.00+"


def _bucket_gap(gap):
    """模型-市场分歧分层：gap 为 abs(model_prob - market_prob)，0~1之间。"""
    if gap is None:
        return None
    if gap < 0.05:
        return "0-5%"
    elif gap < 0.10:
        return "5-10%"
    elif gap < 0.15:
        return "10-15%"
    else:
        return "15%+"


def _bucket_elo(elo_diff):
    """Elo差距分层：elo_diff 为 abs(home_elo - away_elo)。"""
    if elo_diff is None:
        return None
    if elo_diff < 80:
        return "0-80"
    elif elo_diff < 180:
        return "80-180"
    elif elo_diff < 300:
        return "180-300"
    else:
        return "300+"


def build_strategy_a_diagnostics_df(user_name: str, conn) -> pd.DataFrame:
    """
    构建策略A（Top EV>5%）Diagnostics 用的统一底表，4张诊断表都从这个
    DataFrame groupby 生成，不为每张表单独重写聚合逻辑。

    只消费 compute_value_lab_results() 已经算好的策略A下注明细（不重新
    计算AI概率，不修改策略A/B/C定义），额外补充：
      - model_prob / market_prob / prob_gap —— 来自决策快照，
        _simulate_strategy 已经把 model_prob/market_prob 透传进每笔bet。
      - home_elo / away_elo / elo_diff_abs —— 来自 SQLite matches 表，
        一次性批量查询（而非逐笔查询），查不到时这几个字段为 None，
        不影响该行在其余3张表中的统计。
      - odds_bucket / gap_bucket / elo_bucket —— 预先分好类，供聚合
        函数直接 groupby，避免每张表重复写分层判断。

    返回：pandas.DataFrame。策略A没有产生任何下注时返回空 DataFrame
    （调用方需要用 df.empty 判断）。
    """
    data = compute_value_lab_results(user_name, conn)
    bets = data["strategies"]["A"]["bets"]

    if not bets:
        return pd.DataFrame()

    # 批量查一次 Elo（而不是每笔下注单独查一次 SQLite）。
    # get_elo_map_for_matches 是本次新增的小 helper（见 database/db_utils.py），
    # 查询失败或函数不存在时优雅降级为空字典，Elo 相关字段全部为 None，
    # 不影响其余诊断表。
    match_ids = [b["match_id"] for b in bets]
    try:
        from database.db_utils import get_elo_map_for_matches  # 延迟导入，避免模块级循环依赖
        elo_map = get_elo_map_for_matches(conn, match_ids)
    except Exception:
        elo_map = {}

    rows = []
    for b in bets:
        model_prob = b.get("model_prob")
        market_prob = b.get("market_prob")
        prob_gap = (
            abs(model_prob - market_prob)
            if model_prob is not None and market_prob is not None
            else None
        )

        try:
            lookup_id = int(b["match_id"])
        except (TypeError, ValueError):
            lookup_id = b["match_id"]
        elo_entry = elo_map.get(lookup_id, {})
        home_elo = elo_entry.get("home_elo")
        away_elo = elo_entry.get("away_elo")
        elo_diff_abs = (
            abs(home_elo - away_elo)
            if home_elo is not None and away_elo is not None
            else None
        )

        rows.append({
            "match_id": b["match_id"],
            "home_team": b["home_team"],
            "away_team": b["away_team"],
            "date": b["date"],
            "bet_type": b["outcome"],
            "ev": b["ev"],
            "odds": b["odds"],
            "hit": b["hit"],
            "profit": b["profit"],
            "model_prob": model_prob,
            "market_prob": market_prob,
            "prob_gap": prob_gap,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff_abs": elo_diff_abs,
            "is_draw_or_longshot": (b["outcome"] == "draw") or (b["odds"] >= 4.0),
            "odds_bucket": _bucket_odds(b["odds"]),
            "gap_bucket": _bucket_gap(prob_gap),
            "elo_bucket": _bucket_elo(elo_diff_abs),
        })

    return pd.DataFrame(rows)


def _agg_group_stats(group: pd.DataFrame) -> dict:
    """
    单个分组的通用聚合指标：场次 / 命中率 / ROI / 平均EV / 平均赔率 / 最大回撤。
    统计口径与 shared_utils._simulate_strategy 完全一致：
      - 命中率 = sum(hit) / n_bets
      - ROI = sum(profit) / n_bets * 100（百分比数值）
      - 最大回撤：按 date 排序后，对该分组内的累计 profit 曲线取
        peak-to-trough 最大值。
    """
    n = len(group)
    if n == 0:
        return {
            "n_bets": 0, "hit_rate": None, "roi": None,
            "avg_ev": None, "avg_odds": None, "max_drawdown": 0.0,
        }

    hit_rate = float(group["hit"].mean())
    roi = float(group["profit"].sum() / n * 100)
    avg_ev = float(group["ev"].mean())
    avg_odds = float(group["odds"].mean())

    max_dd = 0.0
    cum = 0.0
    peak = 0.0
    for p in group.sort_values("date")["profit"]:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    return {
        "n_bets": n, "hit_rate": hit_rate, "roi": roi,
        "avg_ev": avg_ev, "avg_odds": avg_odds, "max_drawdown": max_dd,
    }


def _diagnose_by_bet_type(df: pd.DataFrame) -> dict:
    """诊断表1：策略A按主胜/平局/客胜拆分。"""
    rows = []
    stats_by_type = {}
    for bt in BET_TYPE_ORDER:
        sub = df[df["bet_type"] == bt]
        if sub.empty:
            continue
        stats = _agg_group_stats(sub)
        stats_by_type[bt] = stats
        rows.append({
            "下注类型": BET_TYPE_LABEL[bt],
            "场次": stats["n_bets"],
            "命中率": stats["hit_rate"],
            "ROI": stats["roi"],
            "平均EV": stats["avg_ev"],
            "平均赔率": stats["avg_odds"],
            "最大回撤": stats["max_drawdown"],
        })

    conclusion = None
    insufficient_sample = False
    if "draw" in stats_by_type and len(stats_by_type) >= 2:
        draw_stats = stats_by_type["draw"]
        other_rois = [s["roi"] for k, s in stats_by_type.items() if k != "draw" and s["roi"] is not None]
        other_hits = [s["hit_rate"] for k, s in stats_by_type.items() if k != "draw" and s["hit_rate"] is not None]
        if (draw_stats["roi"] is not None and other_rois and all(draw_stats["roi"] < r for r in other_rois)
                and draw_stats["hit_rate"] is not None and other_hits and all(draw_stats["hit_rate"] < h for h in other_hits)):
            conclusion = "平局EV是主要污染源"
            insufficient_sample = draw_stats["n_bets"] < DIAGNOSTICS_MIN_SAMPLE_SIZE

    return {"rows": rows, "conclusion": conclusion, "insufficient_sample": insufficient_sample}


def _diagnose_by_odds_bucket(df: pd.DataFrame) -> dict:
    """诊断表2：按市场赔率区间分层。"""
    rows = []
    stats_by_bucket = {}
    for bucket in ODDS_BUCKET_ORDER:
        sub = df[df["odds_bucket"] == bucket]
        if sub.empty:
            continue
        stats = _agg_group_stats(sub)
        stats_by_bucket[bucket] = stats
        implied_prob = float((1 / sub["odds"]).mean())
        model_avg_prob = float(sub["model_prob"].mean()) if sub["model_prob"].notna().any() else None
        rows.append({
            "赔率区间": bucket,
            "下注场次": stats["n_bets"],
            "命中率": stats["hit_rate"],
            "ROI": stats["roi"],
            "平均EV": stats["avg_ev"],
            "市场隐含胜率": implied_prob,
            "模型平均预测概率": model_avg_prob,
        })

    conclusion = None
    insufficient_sample = False
    if "4.00+" in stats_by_bucket and len(stats_by_bucket) >= 2:
        worst = stats_by_bucket["4.00+"]
        other_rois = [s["roi"] for k, s in stats_by_bucket.items() if k != "4.00+" and s["roi"] is not None]
        if worst["roi"] is not None and other_rois and all(worst["roi"] < r for r in other_rois):
            conclusion = "模型在冷门方向容易产生假EV"
            insufficient_sample = worst["n_bets"] < DIAGNOSTICS_MIN_SAMPLE_SIZE

    return {"rows": rows, "conclusion": conclusion, "insufficient_sample": insufficient_sample}


def _diagnose_by_gap_bucket(df: pd.DataFrame) -> dict:
    """诊断表3：按模型-市场分歧大小分层。"""
    rows = []
    stats_by_bucket = {}
    for bucket in GAP_BUCKET_ORDER:
        sub = df[df["gap_bucket"] == bucket]
        if sub.empty:
            continue
        stats = _agg_group_stats(sub)
        stats_by_bucket[bucket] = stats
        draw_share = float((sub["bet_type"] == "draw").mean())
        longshot_share = float((sub["odds"] >= 4.0).mean())
        rows.append({
            "分歧区间": bucket,
            "下注场次": stats["n_bets"],
            "命中率": stats["hit_rate"],
            "ROI": stats["roi"],
            "平均EV": stats["avg_ev"],
            "平局下注占比": draw_share,
            "冷门赔率占比": longshot_share,
        })

    conclusion = None
    insufficient_sample = False
    if "15%+" in stats_by_bucket and len(stats_by_bucket) >= 2:
        worst = stats_by_bucket["15%+"]
        other_rois = [s["roi"] for k, s in stats_by_bucket.items() if k != "15%+" and s["roi"] is not None]
        if worst["roi"] is not None and other_rois and all(worst["roi"] < r for r in other_rois):
            conclusion = "高分歧区域存在较多假EV"
            insufficient_sample = worst["n_bets"] < DIAGNOSTICS_MIN_SAMPLE_SIZE

    return {"rows": rows, "conclusion": conclusion, "insufficient_sample": insufficient_sample}


def _diagnose_by_elo_bucket(df: pd.DataFrame) -> dict:
    """
    诊断表4：按Elo差距分层。
    只使用 elo_bucket 非空的行；Elo 整体不可用时（SQLite 查不到任何
    match_id 的 home_elo/away_elo），返回 elo_unavailable=True，
    由 value_lab.py 展示提示文案而不是空表格。
    """
    sub_all = df[df["elo_bucket"].notna()]
    if sub_all.empty:
        return {"rows": [], "conclusion": None, "insufficient_sample": False, "elo_unavailable": True}

    rows = []
    stats_by_bucket = {}
    for bucket in ELO_BUCKET_ORDER:
        sub = sub_all[sub_all["elo_bucket"] == bucket]
        if sub.empty:
            continue
        stats = _agg_group_stats(sub)
        stats_by_bucket[bucket] = stats
        draw_share = float((sub["bet_type"] == "draw").mean())
        avg_gap = float(sub["prob_gap"].mean()) if sub["prob_gap"].notna().any() else None
        rows.append({
            "Elo差距": bucket,
            "下注场次": stats["n_bets"],
            "命中率": stats["hit_rate"],
            "ROI": stats["roi"],
            "平局EV占比": draw_share,
            "模型-市场平均分歧": avg_gap,
        })

    conclusion = None
    insufficient_sample = False
    if "300+" in stats_by_bucket:
        worst = stats_by_bucket["300+"]
        worst_draw_share = next((r["平局EV占比"] for r in rows if r["Elo差距"] == "300+"), None)
        other_rois = [s["roi"] for k, s in stats_by_bucket.items() if k != "300+" and s["roi"] is not None]
        # "占比高"取 >30% 作为判定门槛（无官方口径可依据时的合理默认值，
        # 如需调整敏感度，改这一个数字即可）。
        if (worst_draw_share is not None and worst_draw_share > 0.3
                and worst["roi"] is not None and other_rois and all(worst["roi"] < r for r in other_rois)):
            conclusion = "强弱悬殊比赛中的平局EV可能是系统性偏差"
            insufficient_sample = worst["n_bets"] < DIAGNOSTICS_MIN_SAMPLE_SIZE

    return {"rows": rows, "conclusion": conclusion, "insufficient_sample": insufficient_sample, "elo_unavailable": False}


def compute_strategy_a_diagnostics(user_name: str, conn) -> dict:
    """
    Value Lab Diagnostics 区块的唯一计算入口，value_lab.py 只负责渲染，
    不在页面文件里做任何聚合计算。

    返回：{
        "n_bets": int,             # 策略A总下注场次
        "elo_available": bool,     # 是否至少有一场下注查到了 Elo 数据
        "tables": {
            "by_bet_type":    {"rows": [...], "conclusion": str|None, "insufficient_sample": bool},
            "by_odds_bucket": {同上},
            "by_gap_bucket":  {同上},
            "by_elo_bucket":  {同上 + "elo_unavailable": bool},
        },
    }
    """
    df = build_strategy_a_diagnostics_df(user_name, conn)

    if df.empty:
        empty_table = {"rows": [], "conclusion": None, "insufficient_sample": False}
        return {
            "n_bets": 0,
            "elo_available": False,
            "tables": {
                "by_bet_type": empty_table,
                "by_odds_bucket": empty_table,
                "by_gap_bucket": empty_table,
                "by_elo_bucket": {**empty_table, "elo_unavailable": True},
            },
        }

    elo_available = bool(df["elo_diff_abs"].notna().any())

    return {
        "n_bets": len(df),
        "elo_available": elo_available,
        "tables": {
            "by_bet_type": _diagnose_by_bet_type(df),
            "by_odds_bucket": _diagnose_by_odds_bucket(df),
            "by_gap_bucket": _diagnose_by_gap_bucket(df),
            "by_elo_bucket": _diagnose_by_elo_bucket(df),
        },
    }
