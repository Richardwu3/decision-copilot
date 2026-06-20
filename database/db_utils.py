"""
db_utils.py
===========
数据库写入层。仅负责把模型/特征/用户决策/比赛结果写入 SQLite，
不包含任何建模逻辑，不修改 worldcup_predictor_v2.py 的计算流程。

设计原则：
- 所有函数接收"已经算好的值"作为参数，调用方（worldcup_predictor_v2.py 或 app.py）
  负责从自己的变量里取值传进来，本文件不重新计算任何特征或预测。
- 字段名严格对应 init_db.py 中定义的表结构。
- 注意 prediction_features 表字段名为 value_ratio（不是 log_value_ratio），
  因为这是数据库的统一命名约定；写入时由调用方把
  worldcup_predictor_v2.py 里的 log_value_ratio 值传给 value_ratio 参数即可，
  不改变该值本身的含义或计算方式。
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "decision_copilot.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """返回一个允许跨线程使用（Streamlit 场景）的数据库连接。"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# 1. matches 表
# ============================================================

def save_match(conn: sqlite3.Connection,
              match_id: int,
              date,
              home_team: str,
              away_team: str,
              venue: str,
              home_adv: int,
              home_elo: float,
              away_elo: float,
              home_value: float,
              away_value: float) -> None:
    """
    写入/更新一场比赛的静态信息。
    使用 INSERT OR REPLACE，按 match_id 幂等写入。
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO matches
            (match_id, date, home_team, away_team, venue,
             home_adv, home_elo, away_elo, home_value, away_value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (match_id, str(date), home_team, away_team, venue,
         int(home_adv), float(home_elo), float(away_elo),
         float(home_value), float(away_value))
    )
    conn.commit()


# ============================================================
# 2. predictions 表
# ============================================================

def save_prediction(conn: sqlite3.Connection,
                    match_id: int,
                    model_version: str,
                    prob_home: float,
                    prob_draw: float,
                    prob_away: float,
                    lambda_home: float,
                    lambda_away: float,
                    pred_date=None) -> int:
    """
    写入一次模型预测结果，返回新插入行的 pred_id。
    每次调用都会新增一条记录（同一场比赛允许有多个历史预测版本）。
    """
    pred_date = pred_date or datetime.now()
    cur = conn.execute(
        """
        INSERT INTO predictions
            (match_id, pred_date, model_version,
             prob_home, prob_draw, prob_away, lambda_home, lambda_away)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (match_id, str(pred_date), model_version,
         float(prob_home), float(prob_draw), float(prob_away),
         float(lambda_home), float(lambda_away))
    )
    conn.commit()
    return cur.lastrowid


# ============================================================
# 3. prediction_features 表
# ============================================================

def save_prediction_features(conn: sqlite3.Connection,
                              match_id: int,
                              elo_diff: float,
                              value_ratio: float,
                              weighted_xg_diff: float,
                              weighted_xga_diff: float,
                              home_adv: int,
                              injury_impact_home: float,
                              injury_impact_away: float,
                              pred_date=None) -> int:
    """
    写入某次预测时使用的特征上下文，用于赛后复盘与特征归因。

    参数对应关系（来自 worldcup_predictor_v2.py 的 FEATURE_COLS / 局部变量）：
        elo_diff            <- elo_diff
        value_ratio         <- log_value_ratio   （字段改名，值不变）
        weighted_xg_diff    <- weighted_xg_diff
        weighted_xga_diff   <- weighted_xga_diff
        home_adv            <- home_adv
        injury_impact_home  <- inj_h
        injury_impact_away  <- inj_a

    注：weighted_res_diff（xG 残差特征）当前数据库表未单独建列，
        如需持久化可在 Phase 2 扩展 prediction_features 表。
    """
    pred_date = pred_date or datetime.now()
    cur = conn.execute(
        """
        INSERT INTO prediction_features
            (match_id, pred_date, elo_diff, value_ratio,
             weighted_xg_diff, weighted_xga_diff, home_adv,
             injury_impact_home, injury_impact_away)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (match_id, str(pred_date), float(elo_diff), float(value_ratio),
         float(weighted_xg_diff), float(weighted_xga_diff), int(home_adv),
         float(injury_impact_home), float(injury_impact_away))
    )
    conn.commit()
    return cur.lastrowid


# ============================================================
# 4. decision_events 表
# ============================================================

def save_decision_event(conn: sqlite3.Connection,
                        match_id: int,
                        user_name: str,
                        event_type: str,
                        choice: str = None,
                        confidence: int = None,
                        reason_tags: str = None,
                        reason_detail: str = None,
                        changed_from: str = None,
                        timestamp=None) -> int:
    """
    写入一条用户决策事件。event_type 建议取值：
        'decision_made'      首次做出决策
        'decision_changed'   修改先前决策（changed_from 记录原选择）
        'review_completed'   赛后复盘完成
    """
    timestamp = timestamp or datetime.now()
    cur = conn.execute(
        """
        INSERT INTO decision_events
            (match_id, user_name, timestamp, event_type,
             choice, confidence, reason_tags, reason_detail, changed_from)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (match_id, user_name, str(timestamp), event_type,
         choice, confidence, reason_tags, reason_detail, changed_from)
    )
    conn.commit()
    return cur.lastrowid


def get_latest_decision(conn: sqlite3.Connection, match_id: int, user_name: str):
    """
    取某用户对某场比赛的最新决策事件（按时间倒序取第一条）。
    返回 sqlite3.Row 或 None。
    """
    cur = conn.execute(
        """
        SELECT * FROM decision_events
        WHERE match_id = ? AND user_name = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (match_id, user_name)
    )
    return cur.fetchone()


def get_decision_history(conn: sqlite3.Connection, match_id: int, user_name: str):
    """取某用户对某场比赛的全部决策事件，按时间正序排列（用于复盘时间线）。"""
    cur = conn.execute(
        """
        SELECT * FROM decision_events
        WHERE match_id = ? AND user_name = ?
        ORDER BY timestamp ASC
        """,
        (match_id, user_name)
    )
    return cur.fetchall()


# ============================================================
# 5. results 表
# ============================================================

def save_result(conn: sqlite3.Connection,
                match_id: int,
                home_goals: int,
                away_goals: int,
                is_finished: bool = True) -> None:
    """
    写入/更新比赛结果。result 字段根据进球数自动计算（'H' / 'D' / 'A'）。
    """
    if home_goals > away_goals:
        result = "H"
    elif home_goals < away_goals:
        result = "A"
    else:
        result = "D"

    conn.execute(
        """
        INSERT OR REPLACE INTO results
            (match_id, home_goals, away_goals, result, is_finished)
        VALUES (?, ?, ?, ?, ?)
        """,
        (match_id, int(home_goals), int(away_goals), result, int(is_finished))
    )
    conn.commit()


# ============================================================
# 6. 组合读取函数（供 Dashboard / Match Detail 使用）
# ============================================================

def get_all_matches_with_latest_prediction(conn: sqlite3.Connection):
    """
    Dashboard 用：返回所有比赛 + 各自最新一次预测（按 pred_date 取最新）。
    """
    cur = conn.execute(
        """
        SELECT m.*,
               p.prob_home, p.prob_draw, p.prob_away,
               p.lambda_home, p.lambda_away, p.pred_date, p.model_version
        FROM matches m
        LEFT JOIN predictions p
            ON p.match_id = m.match_id
            AND p.pred_id = (
                SELECT pred_id FROM predictions
                WHERE match_id = m.match_id
                ORDER BY pred_date DESC LIMIT 1
            )
        ORDER BY m.date ASC, m.match_id ASC
        """
    )
    return cur.fetchall()


def get_match_full_context(conn: sqlite3.Connection, match_id: int):
    """
    Match Detail 用：一次性取出某场比赛的全部相关数据
    （比赛信息、最新预测、最新特征、结果）。
    返回 dict，各 key 对应一张表的最新一行（无数据则为 None）。
    """
    match_row = conn.execute(
        "SELECT * FROM matches WHERE match_id = ?", (match_id,)
    ).fetchone()

    pred_row = conn.execute(
        """
        SELECT * FROM predictions
        WHERE match_id = ? ORDER BY pred_date DESC LIMIT 1
        """, (match_id,)
    ).fetchone()

    feat_row = conn.execute(
        """
        SELECT * FROM prediction_features
        WHERE match_id = ? ORDER BY pred_date DESC LIMIT 1
        """, (match_id,)
    ).fetchone()

    result_row = conn.execute(
        "SELECT * FROM results WHERE match_id = ?", (match_id,)
    ).fetchone()

    return {
        "match": match_row,
        "prediction": pred_row,
        "features": feat_row,
        "result": result_row,
    }
