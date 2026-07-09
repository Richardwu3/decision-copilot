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

时间感知升级（本次新增）：
- predictions 表新增 6 列，用于记录"这次预测是在哪个 prediction_date 下，
  用了历史快照还是最新快照"：prediction_date / elo_source /
  elo_home_used / elo_away_used / value_home_used / value_away_used。
- 已有数据库如果缺这些列，ensure_database_schema() 会用 ALTER TABLE
  ADD COLUMN 就地补全（不会删除/重建数据库，见该函数文档）。
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
        # 时间感知升级新增列：
        "prediction_date", "elo_source",
        "elo_home_used", "elo_away_used",
        "value_home_used", "value_away_used",
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

# ============================================================
# 可安全"就地新增"的列（ALTER TABLE ADD COLUMN，不删除任何数据）。
# 只有出现在这里的列，在旧数据库上缺失时会被自动 ADD COLUMN 补全；
# EXPECTED_SCHEMA 中出现但不在这里的列若缺失，说明是比这次升级更早、
# 更根本的结构缺失（旧版本 schema 不兼容），仍然走原有的
# "删除并重建"兜底逻辑（见 ensure_database_schema）。
# 类型字符串会被直接拼进 "ALTER TABLE t ADD COLUMN col <TYPE>"，
# 因此只能用 SQLite 支持"非常量表达式以外"的 DEFAULT（字面量/NULL）。
# ============================================================
MIGRATION_ADDABLE_COLUMNS = {
    "predictions": {
        "prediction_date":  "DATE",
        "elo_source":       "TEXT DEFAULT 'unknown'",
        "elo_home_used":    "REAL",
        "elo_away_used":    "REAL",
        "value_home_used":  "REAL",
        "value_away_used":  "REAL",
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
    （这正是 Streamlit Cloud 报错的根因：sqlite3.connect() 在文件
    不存在时会静默创建一个 0 张表的空文件，旧的判断逻辑"文件存在就跳过初始化"
    完全检测不出这种情况）。

    检查策略（三层，按"越不破坏数据越优先"排序）：
      1. 轻量修复：任何期望的表不存在 -> 执行 CREATE TABLE IF NOT EXISTS
         （幂等无害，可以放心在每次应用启动时都跑一遍）。
      2. 就地迁移（本次新增，替代了旧版本"缺列就删库重建"的行为）：
         表存在但缺列时，若缺失的列都在 MIGRATION_ADDABLE_COLUMNS 里
         登记过（当前仅 predictions 表的时间感知新增列），逐列执行
         ALTER TABLE ADD COLUMN 就地补全 —— 不删除、不清空任何已有数据；
         SQLite 对带字面量 DEFAULT 的 ADD COLUMN 会自动把该默认值回填到
         所有已有行（因此旧记录的 elo_source 会自动变成 'unknown'，
         符合"向后兼容默认值"的要求）。
      3. 重型修复（兜底，仅在缺失列不在可迁移清单内时触发）：
         说明这是比本次升级更早、更根本的结构不兼容（例如核心列缺失），
         此时才退回到"删除数据库文件、重新 init_database()"的旧行为。
         正常升级路径（本次新增列）不会走到这一步。

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
        if not missing_cols:
            continue

        addable = set(MIGRATION_ADDABLE_COLUMNS.get(table, {}).keys())
        safely_migratable = missing_cols & addable
        unmigratable = missing_cols - addable

        if safely_migratable:
            print(f"[ensure_database_schema] 表 {table} 缺失可就地迁移的列："
                  f"{safely_migratable}，执行 ALTER TABLE ADD COLUMN（不影响已有数据）。")
            for col in sorted(safely_migratable):
                col_type = MIGRATION_ADDABLE_COLUMNS[table][col]
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                    conn.commit()
                    print(f"    + {table}.{col} {col_type} 已添加。")
                except sqlite3.OperationalError as e:
                    # 例如并发场景下列已被其他进程加上，忽略"duplicate column"类错误
                    print(f"    ! 添加 {table}.{col} 时出现异常（可能已存在，忽略）：{e}")

        if unmigratable:
            schema_broken = True
            print(f"[ensure_database_schema] 表 {table} 缺失列：{unmigratable}，"
                  f"不在可就地迁移清单内，判定为结构不兼容的旧版本 schema。")

    if schema_broken:
        print(f"[ensure_database_schema] 数据库结构与当前代码不兼容（且无法就地迁移），"
              f"删除并重建：{db_path}")
        conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        init_database(db_path)
        conn = get_connection(db_path)
        print(f"[ensure_database_schema] 数据库已重建完成。")
    else:
        print(f"[ensure_database_schema] 数据库 schema 校验通过（含就地迁移），共 "
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
                    pred_date=None,
                    prediction_date=None,
                    elo_source: str = "unknown",
                    elo_home_used: float = None,
                    elo_away_used: float = None,
                    value_home_used: float = None,
                    value_away_used: float = None) -> int:
    """
    写入一次模型预测结果，返回新插入行的 pred_id。
    每次调用都会新增一条记录（同一场比赛允许有多个历史预测版本）。

    时间感知升级新增参数（均为可选，向后兼容旧调用方）：
      prediction_date  — 生成这次预测时使用的 prediction_date（显式基准日，
                          与 pred_date/写入时间戳不同：pred_date 是"这条记录
                          何时被写入数据库"，prediction_date 是"预测逻辑上
                          假设自己站在哪一天"，两者在离线批量回测时可能不同）。
      elo_source        — "historical" | "latest" | "latest_fallback" | "unknown"
      elo_home_used / elo_away_used     — 本次预测实际使用的主/客队 Elo
      value_home_used / value_away_used — 本次预测实际使用的主/客队身价
    旧调用方不传这些参数时，写入 NULL / 'unknown'，不影响任何现有调用。
    """
    pred_date = pred_date or datetime.now()
    cur = conn.execute(
        """
        INSERT INTO predictions
            (match_id, pred_date, model_version,
             prob_home, prob_draw, prob_away, lambda_home, lambda_away,
             prediction_date, elo_source,
             elo_home_used, elo_away_used, value_home_used, value_away_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (match_id, str(pred_date), model_version,
         float(prob_home), float(prob_draw), float(prob_away),
         float(lambda_home), float(lambda_away),
         str(prediction_date) if prediction_date is not None else None,
         elo_source,
         float(elo_home_used) if elo_home_used is not None else None,
         float(elo_away_used) if elo_away_used is not None else None,
         float(value_home_used) if value_home_used is not None else None,
         float(value_away_used) if value_away_used is not None else None)
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


def get_elo_map_for_matches(conn: sqlite3.Connection, match_ids: list) -> dict:
    """
    Value Lab Diagnostics 用：一次性批量查询多场比赛的 Elo 快照，
    避免逐场调用 get_match_full_context / 逐场查 SQLite
    （例如 96 场比赛的场景下，从 96 次查询降为 1 次）。

    match_ids 可以混杂字符串/整数（Google Sheet 里的 match_id 是字符串，
    SQLite 主键是整数），本函数内部统一转成 int 再查询，转换失败的
    id 会被静默跳过（不会导致整体查询失败）。

    返回：{match_id(int): {"home_elo": float|None, "away_elo": float|None}}
    match_ids 为空、查询失败、或该批比赛在 matches 表里都没有 Elo 时，
    返回空字典（调用方据此判断 Elo 数据整体不可用），不抛异常，
    不影响 Value Lab 其余功能正常运行。
    """
    if not match_ids:
        return {}

    normalized_ids = []
    seen = set()
    for mid in match_ids:
        try:
            mid_int = int(mid)
        except (TypeError, ValueError):
            continue
        if mid_int not in seen:
            seen.add(mid_int)
            normalized_ids.append(mid_int)

    if not normalized_ids:
        return {}

    try:
        placeholders = ",".join("?" for _ in normalized_ids)
        cur = conn.execute(
            f"SELECT match_id, home_elo, away_elo FROM matches WHERE match_id IN ({placeholders})",
            normalized_ids,
        )
        result = {}
        for row in cur.fetchall():
            result[row["match_id"]] = {
                "home_elo": row["home_elo"],
                "away_elo": row["away_elo"],
            }
        return result
    except Exception as e:
        print(f"[db_utils.py] 警告：批量查询 Elo 失败（{e}），"
              f"Value Lab Diagnostics 的「按Elo差距分层」诊断表将不可用。")
        return {}
