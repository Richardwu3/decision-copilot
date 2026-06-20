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
