"""
decision_context_builder.py
============================
AI Context Engine —— 所有 AI 功能（AI 复盘 / 用户画像 / 赛前提醒）获取数据
的唯一入口。

硬约束（务必遵守）：
  - 本模块是 AI 功能读取数据的唯一入口：ai_coach.py 以及未来的
    user_profile.py / pre_match_alert.py 一律通过本模块的
    build_decision_context() / build_decision_context_batch() 获取数据，
    不直接 import database.db / database.db_utils，也不直接读取
    Google Sheets 或 SQLite。
  - 对外只暴露标准化的 DecisionContext（单场）。
  - Decision Taxonomy 用纯规则函数实现（_build_taxonomy），不依赖 AI
    自由发挥；新增标签只需要在 _build_taxonomy 里加一段判断，不影响
    其他逻辑。

复用说明（避免重复实现）：
  - 数据库读取全部复用 database.db_utils.get_match_full_context /
    database.db_utils.get_connection，不重新写 SQL。
  - 用户决策记录复用 shared_utils.get_cached_user_history /
    build_user_decision_map（与 Dashboard / Review Center / Value Lab
    同一套 session 缓存与聚合口径，不产生第二套"最新决策"的判定标准）。
  - AI选择/置信度口径复用 shared_utils.get_ai_choice_and_confidence，
    与 save_decision 写入快照时的口径完全一致。
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any


# ============================================================
# 数据结构定义（DecisionContext 及其子结构）
# ============================================================

@dataclass
class MatchInfo:
    match_id: str
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    date: Optional[str] = None
    venue: Optional[str] = None
    stage: Optional[str] = None  # "group_stage" / "group_stage_decisive" / "knockout"


@dataclass
class UserDecision:
    choice: Optional[str] = None          # 'home' / 'draw' / 'away'
    confidence: Optional[int] = None      # 1-5
    reason: Optional[str] = None
    comment: Optional[str] = None
    predict_home: Optional[int] = None
    predict_away: Optional[int] = None
    predict_btts: Optional[bool] = None
    predict_over25: Optional[bool] = None
    timestamp: Optional[str] = None
    has_decision: bool = False


@dataclass
class AISnapshot:
    prob_home: Optional[float] = None
    prob_draw: Optional[float] = None
    prob_away: Optional[float] = None
    lambda_home: Optional[float] = None
    lambda_away: Optional[float] = None
    btts_prob: Optional[float] = None
    over25_prob: Optional[float] = None
    top_score_1: Optional[str] = None
    top_score_1_prob: Optional[float] = None
    top_score_2: Optional[str] = None
    top_score_2_prob: Optional[float] = None
    ai_choice: Optional[str] = None
    ai_confidence: Optional[float] = None
    model_version: Optional[str] = None
    available: bool = False


@dataclass
class MarketSnapshot:
    prob_home: Optional[float] = None
    prob_draw: Optional[float] = None
    prob_away: Optional[float] = None
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None
    market_choice: Optional[str] = None
    available: bool = False


@dataclass
class MatchResult:
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    result_code: Optional[str] = None   # 'H' / 'D' / 'A'
    is_finished: bool = False


@dataclass
class EVSnapshot:
    ev_home: Optional[float] = None
    ev_draw: Optional[float] = None
    ev_away: Optional[float] = None
    ev_for_user_choice: Optional[float] = None
    available: bool = False


@dataclass
class DecisionContextMetadata:
    model_version: Optional[str] = None
    context_version: str = "v1"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # 各快照是否具备完整数据，供下游 AI 功能判断"信息是否足够生成分析"，
    # 不需要重新逐字段判空。
    data_completeness: Dict[str, bool] = field(default_factory=dict)


@dataclass
class DecisionContext:
    match: MatchInfo
    user_decision: UserDecision
    ai_snapshot: AISnapshot
    market_snapshot: MarketSnapshot
    result: MatchResult
    ev_snapshot: EVSnapshot
    decision_taxonomy: List[str]
    metadata: DecisionContextMetadata

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 工具：安全类型转换 / 安全读取
# ============================================================
# 独立实现（而不是 import shared_utils._to_float / app._safe_row_get），
# 避免本模块反向依赖上层页面文件的私有函数，保持"数据入口"层的独立性。

def _to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_bool(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "是", "yes"):
        return True
    if s in ("false", "0", "否", "no"):
        return False
    return None


def _safe_row_get(row, key, default=None):
    """安全读取 sqlite3.Row 或 dict，行为与 app.py 中的 _safe_row_get 一致。"""
    if row is None:
        return default
    try:
        if hasattr(row, "keys"):
            return row[key] if key in row.keys() else default
        return row.get(key, default)
    except Exception:
        return default


# ============================================================
# 比赛阶段判定（match_id 分段规则，来自项目世界杯赛程约定）
# ============================================================

def _resolve_stage(match_id) -> Optional[str]:
    try:
        mid = int(match_id)
    except (TypeError, ValueError):
        return None
    if mid <= 48:
        return "group_stage"
    elif mid <= 72:
        return "group_stage_decisive"
    else:
        return "knockout"


def _elo_diff(match_row) -> Optional[float]:
    home_elo = _safe_row_get(match_row, "home_elo")
    away_elo = _safe_row_get(match_row, "away_elo")
    if home_elo is None or away_elo is None:
        return None
    return float(home_elo) - float(away_elo)


# ============================================================
# Decision Taxonomy —— 规则驱动，不依赖 AI 自由发挥
# ============================================================
#
# 标签列表与判断规则（新增标签只需在函数体内追加一段 if，不影响其他标签）：
#
#   比赛阶段：      group_stage / group_stage_decisive / knockout
#   实力差距：      strength_gap（|elo_diff|>=200）/ close_match（|elo_diff|<=100）
#                   home_team_strong（elo_diff>=120）/ away_team_strong（elo_diff<=-120）
#   一致性：        ai_agree / ai_disagree（用户 vs AI）
#                   market_agree / market_disagree（用户 vs 市场）
#                   ai_market_disagree（AI 与市场本身不一致）
#                   contrarian_pick（用户既不跟 AI 也不跟市场）
#   信心水平：      high_confidence_user（用户信心>=4）/ low_confidence_user（<=2）
#                   ai_high_confidence（AI置信度>=55%）/ ai_low_confidence（<40%）
#   EV：            positive_ev_pick（所选结果EV>5%）/ negative_ev_pick（EV<0）
#   选择类型：      draw_pick / favorite_pick（选了Elo更高一方）/ underdog_pick（选了Elo更低一方）
#   赛果（仅已结束比赛）：result_correct / result_incorrect
#
def _build_taxonomy(match_row, match_id, user_decision: UserDecision,
                     ai_snapshot: AISnapshot, market_snapshot: MarketSnapshot,
                     ev_snapshot: EVSnapshot, result: MatchResult) -> List[str]:
    tags = []

    elo_diff = _elo_diff(match_row)
    stage = _resolve_stage(match_id)

    if stage:
        tags.append(stage)

    if elo_diff is not None:
        abs_diff = abs(elo_diff)
        if abs_diff >= 200:
            tags.append("strength_gap")
        if abs_diff <= 100:
            tags.append("close_match")
        if elo_diff >= 120:
            tags.append("home_team_strong")
        if elo_diff <= -120:
            tags.append("away_team_strong")

    user_choice = user_decision.choice
    ai_choice = ai_snapshot.ai_choice
    market_choice = market_snapshot.market_choice

    if user_choice and ai_choice:
        tags.append("ai_agree" if user_choice == ai_choice else "ai_disagree")
    if user_choice and market_choice:
        tags.append("market_agree" if user_choice == market_choice else "market_disagree")
    if ai_choice and market_choice and ai_choice != market_choice:
        tags.append("ai_market_disagree")
    if (user_choice and ai_choice and market_choice
            and user_choice != ai_choice and user_choice != market_choice):
        tags.append("contrarian_pick")

    if user_decision.confidence is not None:
        if user_decision.confidence >= 4:
            tags.append("high_confidence_user")
        elif user_decision.confidence <= 2:
            tags.append("low_confidence_user")

    if ai_snapshot.ai_confidence is not None:
        if ai_snapshot.ai_confidence >= 0.55:
            tags.append("ai_high_confidence")
        elif ai_snapshot.ai_confidence < 0.40:
            tags.append("ai_low_confidence")

    if ev_snapshot.ev_for_user_choice is not None:
        if ev_snapshot.ev_for_user_choice > 0.05:
            tags.append("positive_ev_pick")
        elif ev_snapshot.ev_for_user_choice < 0:
            tags.append("negative_ev_pick")

    if user_choice == "draw":
        tags.append("draw_pick")
    if user_choice and elo_diff is not None:
        favorite = "home" if elo_diff > 0 else ("away" if elo_diff < 0 else None)
        if favorite is not None and user_choice in ("home", "away"):
            tags.append("favorite_pick" if user_choice == favorite else "underdog_pick")

    if result.is_finished and user_choice and result.result_code:
        code_to_choice = {"H": "home", "D": "draw", "A": "away"}
        actual = code_to_choice.get(str(result.result_code).strip().upper())
        if actual is not None:
            tags.append("result_correct" if user_choice == actual else "result_incorrect")

    return tags


# ============================================================
# 主构建函数
# ============================================================

def build_decision_context(user_name: str,
                            match_id,
                            conn=None,
                            match_ctx: Optional[dict] = None,
                            decision_record: Optional[dict] = None) -> DecisionContext:
    """
    组装单场比赛的 DecisionContext —— AI 功能获取数据的唯一入口。

    参数：
      user_name        当前用户昵称。
      match_id          比赛ID。
      conn              SQLite 连接；为 None 时内部自行获取
                        （独立调用场景，例如未来的赛前提醒批处理任务）。
      match_ctx         可选，已经查询好的 get_match_full_context() 返回值。
                        调用方（例如 review.py）通常已经为渲染逐场卡片查询过
                        一次，传入此参数可避免本函数内部重复查询 SQLite。
      decision_record   可选，已经从 build_user_decision_map() 取出的该用户
                        该场比赛的最新决策记录（dict）。传入可避免重复扫描
                        Google Sheets 全量历史。

    返回：DecisionContext。任何环节数据缺失都不会抛异常，缺失字段体现为
         None / available=False，由下游 AI 功能自行判断是否有足够信息。
    """
    from database.db_utils import get_connection, get_match_full_context
    from shared_utils import (
        get_cached_user_history, build_user_decision_map,
        get_ai_choice_and_confidence,
    )

    if conn is None:
        conn = get_connection()

    if match_ctx is None:
        try:
            match_ctx = get_match_full_context(conn, int(match_id))
        except Exception:
            match_ctx = {"match": None, "prediction": None, "features": None, "result": None}

    match_row = match_ctx.get("match")
    pred_row = match_ctx.get("prediction")
    result_row = match_ctx.get("result")

    if decision_record is None:
        user_history = get_cached_user_history(user_name)
        decision_map = build_user_decision_map(user_history)
        decision_record = decision_map.get(str(match_id))

    # ---- MatchInfo ----
    match_info = MatchInfo(
        match_id=str(match_id),
        home_team=_safe_row_get(match_row, "home_team"),
        away_team=_safe_row_get(match_row, "away_team"),
        date=_safe_row_get(match_row, "date"),
        venue=_safe_row_get(match_row, "venue"),
        stage=_resolve_stage(match_id),
    )

    # ---- UserDecision ----
    if decision_record:
        user_decision = UserDecision(
            choice=decision_record.get("choice") or None,
            confidence=_to_int(decision_record.get("confidence")),
            reason=decision_record.get("reason") or None,
            comment=decision_record.get("comment") or None,
            predict_home=_to_int(decision_record.get("predict_home")),
            predict_away=_to_int(decision_record.get("predict_away")),
            predict_btts=_to_bool(decision_record.get("predict_btts")),
            predict_over25=_to_bool(decision_record.get("predict_over25")),
            timestamp=decision_record.get("timestamp") or None,
            has_decision=True,
        )
    else:
        user_decision = UserDecision(has_decision=False)

    # ---- AISnapshot ----
    # 优先使用决策记录里的快照（用户当时看到的口径），SQLite 当前预测仅作
    # 补充/回退，避免"事后用最新模型重算"造成数据穿越（与 Value Lab 的
    # "唯一真相来源"原则一致，见 shared_utils._build_value_lab_items 注释）。
    ai_home = _to_float(decision_record.get("ai_home_prob")) if decision_record else None
    ai_draw = _to_float(decision_record.get("ai_draw_prob")) if decision_record else None
    ai_away = _to_float(decision_record.get("ai_away_prob")) if decision_record else None

    if ai_home is None and pred_row is not None:
        ai_home = _safe_row_get(pred_row, "prob_home")
        ai_draw = _safe_row_get(pred_row, "prob_draw")
        ai_away = _safe_row_get(pred_row, "prob_away")

    ai_choice, ai_confidence = (None, None)
    if ai_home is not None and ai_draw is not None and ai_away is not None:
        ai_choice, ai_confidence = get_ai_choice_and_confidence(ai_home, ai_draw, ai_away)

    ai_snapshot = AISnapshot(
        prob_home=ai_home, prob_draw=ai_draw, prob_away=ai_away,
        lambda_home=_safe_row_get(pred_row, "lambda_home"),
        lambda_away=_safe_row_get(pred_row, "lambda_away"),
        btts_prob=_to_float(decision_record.get("ai_btts_prob")) if decision_record else None,
        over25_prob=_to_float(decision_record.get("ai_over25_prob")) if decision_record else None,
        top_score_1=decision_record.get("ai_top_score_1") if decision_record else None,
        top_score_1_prob=_to_float(decision_record.get("ai_top_score_1_prob")) if decision_record else None,
        top_score_2=decision_record.get("ai_top_score_2") if decision_record else None,
        top_score_2_prob=_to_float(decision_record.get("ai_top_score_2_prob")) if decision_record else None,
        ai_choice=ai_choice,
        ai_confidence=ai_confidence,
        model_version=_safe_row_get(pred_row, "model_version"),
        available=(ai_home is not None and ai_draw is not None and ai_away is not None),
    )

    # ---- MarketSnapshot ----
    market_home = _to_float(decision_record.get("market_home_prob")) if decision_record else None
    market_draw = _to_float(decision_record.get("market_draw_prob")) if decision_record else None
    market_away = _to_float(decision_record.get("market_away_prob")) if decision_record else None
    home_odds = _to_float(decision_record.get("home_odds")) if decision_record else None
    draw_odds = _to_float(decision_record.get("draw_odds")) if decision_record else None
    away_odds = _to_float(decision_record.get("away_odds")) if decision_record else None

    market_choice = None
    if market_home is not None and market_draw is not None and market_away is not None:
        prob_map = {"home": market_home, "draw": market_draw, "away": market_away}
        market_choice = max(prob_map, key=prob_map.get)

    market_snapshot = MarketSnapshot(
        prob_home=market_home, prob_draw=market_draw, prob_away=market_away,
        home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds,
        market_choice=market_choice,
        available=(market_home is not None and market_draw is not None and market_away is not None),
    )

    # ---- MatchResult ----
    is_finished = bool(_safe_row_get(result_row, "is_finished", False))
    result = MatchResult(
        home_goals=_safe_row_get(result_row, "home_goals"),
        away_goals=_safe_row_get(result_row, "away_goals"),
        result_code=_safe_row_get(result_row, "result"),
        is_finished=is_finished,
    )

    # ---- EVSnapshot ----
    ev_home = _to_float(decision_record.get("ev_home")) if decision_record else None
    ev_draw = _to_float(decision_record.get("ev_draw")) if decision_record else None
    ev_away = _to_float(decision_record.get("ev_away")) if decision_record else None

    if None in (ev_home, ev_draw, ev_away) and ai_snapshot.available and market_snapshot.available:
        try:
            ev_home = ai_home / market_home - 1
            ev_draw = ai_draw / market_draw - 1
            ev_away = ai_away / market_away - 1
        except Exception:
            ev_home = ev_draw = ev_away = None

    ev_for_choice = None
    if user_decision.choice in ("home", "draw", "away"):
        ev_for_choice = {"home": ev_home, "draw": ev_draw, "away": ev_away}.get(user_decision.choice)

    ev_snapshot = EVSnapshot(
        ev_home=ev_home, ev_draw=ev_draw, ev_away=ev_away,
        ev_for_user_choice=ev_for_choice,
        available=(ev_home is not None and ev_draw is not None and ev_away is not None),
    )

    # ---- Decision Taxonomy ----
    taxonomy = _build_taxonomy(
        match_row, match_id, user_decision, ai_snapshot, market_snapshot, ev_snapshot, result
    )

    # ---- Metadata ----
    metadata = DecisionContextMetadata(
        model_version=ai_snapshot.model_version,
        data_completeness={
            "user_decision": user_decision.has_decision,
            "ai_snapshot": ai_snapshot.available,
            "market_snapshot": market_snapshot.available,
            "ev_snapshot": ev_snapshot.available,
            "result": result.is_finished,
        },
    )

    return DecisionContext(
        match=match_info,
        user_decision=user_decision,
        ai_snapshot=ai_snapshot,
        market_snapshot=market_snapshot,
        result=result,
        ev_snapshot=ev_snapshot,
        decision_taxonomy=taxonomy,
        metadata=metadata,
    )


# ============================================================
# 预留：批量构建（供未来"用户画像"功能使用；本次不实现调用方）
# ============================================================

def build_decision_context_batch(user_name: str, conn=None,
                                  only_finished: bool = True) -> List[DecisionContext]:
    """
    批量构建该用户全部（或全部已结束）比赛的 DecisionContext 列表。

    预留给未来的"用户画像"功能：对多场 DecisionContext 做聚合统计
    （例如：平局判断准确率、正EV场次的实际命中率、特定 taxonomy 标签下的
    决策质量分布等）。本次不实现调用方，仅提供数据入口，保证未来该功能
    同样只经过本模块获取数据，不需要重新实现一遍数据组装逻辑。
    """
    from database.db_utils import get_connection, get_all_matches_with_latest_prediction, get_match_full_context
    from shared_utils import get_cached_user_history, build_user_decision_map

    if conn is None:
        conn = get_connection()

    matches = get_all_matches_with_latest_prediction(conn)
    user_history = get_cached_user_history(user_name)
    decision_map = build_user_decision_map(user_history)

    contexts = []
    for m in matches:
        mid = str(m["match_id"])
        match_ctx = get_match_full_context(conn, m["match_id"])
        if only_finished:
            result_row = match_ctx.get("result")
            if result_row is None or not result_row["is_finished"]:
                continue
        ctx = build_decision_context(
            user_name=user_name,
            match_id=mid,
            conn=conn,
            match_ctx=match_ctx,
            decision_record=decision_map.get(mid),
        )
        contexts.append(ctx)
    return contexts
