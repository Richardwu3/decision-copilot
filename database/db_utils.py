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

# ============================================================
# 期望的 schema 定义：表名 -> 该表必须存在的列名集合
# 与 init_db.py 中 SCHEMA_SQL 实际创建的表/列保持逐项一致。
# ensure_database_schema() 用这份定义去检查"实际数据库"是否完整，
# print_database_schema() 用它做调试输出时的参照。
# ============================================================
EXPECTED_SCHEMA = {
    "matches": {
        "match_id", "date", "home_team", "away_team", "venue",
        "home_adv", "home_elo", "away_elo", "home_value", "away_value",
    },
    "predictions": {
        "pred_id", "match_id", "pred_date", "model_version",
        "prob_home", "prob_draw", "prob_away", "lambda_home", "lambda_away",
    },
    "prediction_features": {
        "feature_id", "match_id", "pred_date", "elo_diff", "value_ratio",
        "weighted_xg_diff", "weighted_xga_diff", "home_adv",
        "injury_impact_home", "injury_impact_away",
    },
    "decision_events": {
        "event_id", "match_id", "user_name", "timestamp", "event_type",
        "choice", "confidence", "reason_tags", "reason_detail", "changed_from",
    },
    "results": {
        "match_id", "home_goals", "away_goals", "result", "is_finished",
    },
}


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """返回一个允许跨线程使用（Streamlit 场景）的数据库连接。"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# 健壮的 schema 校验与自动修复（不依赖"文件是否存在"）
# ============================================================

def _get_existing_tables(conn: sqlite3.Connection) -> set:
    """返回数据库中当前实际存在的表名集合。"""
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


def _get_existing_columns(conn: sqlite3.Connection, table: str) -> set:
    """返回某张表当前实际存在的列名集合。表不存在时返回空集合。"""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}  # row[1] 是列名


def ensure_database_schema(conn: sqlite3.Connection, db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    保证数据库 schema 完整、可用，不依赖"数据库文件是否存在"这个前提
    （这正是本次 Streamlit Cloud 报错的根因：sqlite3.connect() 在文件
    不存在时会静默创建一个 0 张表的空文件，旧的判断逻辑"文件存在就跳过初始化"
    完全检测不出这种情况）。

    检查策略（两层）：
      1. 轻量修复：任何期望的表不存在 -> 执行 CREATE TABLE IF NOT EXISTS
         （这一步本身幂等无害，可以放心在每次应用启动时都跑一遍）。
      2. 重型修复（兜底）：表存在，但其中任何一个必需列缺失
         -> 说明这是一个结构不兼容的旧版本残留数据库文件，
            直接删除该数据库文件，重新执行完整 init_database()。
         这种情况在 MVP 阶段选择"重建优于自动 ALTER"，因为：
           - 自动推导每一列的 ALTER TABLE ADD COLUMN 语句、处理列类型变化、
             处理删除列（SQLite 早期版本不支持 DROP COLUMN）等情况复杂度高；
           - 当前数据量小，重建成本远低于维护一套通用 migration 引擎的成本。

    返回：处理后的同一个 conn（仍然可以继续使用，未关闭）。
    """
    from database.init_db import init_database, SCHEMA_SQL  # 延迟导入，避免循环依赖

    existing_tables = _get_existing_tables(conn)
    missing_tables = set(EXPECTED_SCHEMA.keys()) - existing_tables

    if missing_tables:
        print(f"[ensure_database_schema] 检测到缺失的表：{missing_tables}，"
              f"执行 CREATE TABLE IF NOT EXISTS 补全。")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        existing_tables = _get_existing_tables(conn)

    # 轻量修复后，再次检查每张已存在的表，列是否完整
    schema_broken = False
    for table, expected_cols in EXPECTED_SCHEMA.items():
        if table not in existing_tables:
            # 理论上不应该再发生（上面刚补全过），但保留判断防止极端竞态
            schema_broken = True
            print(f"[ensure_database_schema] 严重：表 {table} 补全后仍不存在。")
            break

        actual_cols = _get_existing_columns(conn, table)
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            schema_broken = True
            print(f"[ensure_database_schema] 表 {table} 缺失列：{missing_cols}，"
                  f"判定为结构不兼容的旧版本 schema。")

    if schema_broken:
        print(f"[ensure_database_schema] 数据库结构与当前代码不兼容，"
              f"删除并重建：{db_path}")
        conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        init_database(db_path)
        conn = get_connection(db_path)
        print(f"[ensure_database_schema] 数据库已重建完成。")
    else:
        print(f"[ensure_database_schema] 数据库 schema 校验通过，共 "
              f"{len(EXPECTED_SCHEMA)} 张表均完整。")

    return conn


# ============================================================
# 调试工具
# ============================================================

def print_database_schema(conn: sqlite3.Connection) -> None:
    """
    输出当前数据库的完整 schema 信息，用于在 Streamlit Cloud 等
    远程环境快速定位"实际数据库状态"与"代码期望状态"是否一致。

    输出内容：
      - SQLite 版本号
      - 所有表的名称
      - 每张表的所有字段（名称 + 类型）
    """
    print("=" * 60)
    print("数据库 Schema 调试信息")
    print("=" * 60)

    cur = conn.execute("SELECT sqlite_version()")
    version = cur.fetchone()[0]
    print(f"SQLite 版本：{version}")

    tables = sorted(_get_existing_tables(conn))
    print(f"共 {len(tables)} 张表：{tables}")
    print()

    for table in tables:
        print(f"-- 表：{table}")
        cur = conn.execute(f"PRAGMA table_info({table})")
        columns = cur.fetchall()
        for col in columns:
            # PRAGMA table_info 返回: (cid, name, type, notnull, dflt_value, pk)
            cid, name, col_type, notnull, dflt_value, pk = col
            pk_marker = " [PK]" if pk else ""
            print(f"    {name:<22} {col_type:<12}{pk_marker}")
        print()

    print("=" * 60)


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
