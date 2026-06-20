"""
init_db.py
==========
初始化 Decision Review Copilot 的 SQLite 数据库（Phase 1，5 张表）。

运行方式：
    python database/init_db.py

幂等：重复运行不会破坏已有数据（CREATE TABLE IF NOT EXISTS）。
若需要完全重建，删除 decision_copilot.db 文件后重新运行即可。
"""

import sqlite3
import os

# 数据库文件路径：与本脚本同级目录下的上一级（项目根目录）
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "decision_copilot.db")


SCHEMA_SQL = """
-- 1. 比赛信息表：每场比赛的静态背景信息
CREATE TABLE IF NOT EXISTS matches (
    match_id    INTEGER PRIMARY KEY,
    date        DATE,
    home_team   TEXT,
    away_team   TEXT,
    venue       TEXT,
    home_adv    INTEGER,
    home_elo    INTEGER,
    away_elo    INTEGER,
    home_value  REAL,
    away_value  REAL
);

-- 2. 模型预测表：每次模型运行产生的核心预测结果
CREATE TABLE IF NOT EXISTS predictions (
    pred_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id       INTEGER,
    pred_date      TIMESTAMP,
    model_version  TEXT,
    prob_home      REAL,
    prob_draw      REAL,
    prob_away      REAL,
    lambda_home    REAL,
    lambda_away    REAL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

-- 3. 预测特征表：记录决策时模型看到的上下文（用于赛后复盘 / 特征归因）
CREATE TABLE IF NOT EXISTS prediction_features (
    feature_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id             INTEGER,
    pred_date            TIMESTAMP,
    elo_diff             REAL,
    value_ratio          REAL,
    weighted_xg_diff     REAL,
    weighted_xga_diff    REAL,
    home_adv             INTEGER,
    injury_impact_home   REAL,
    injury_impact_away   REAL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

-- 4. 决策事件流表：用户在 UI 上产生的所有决策行为（核心产品数据）
CREATE TABLE IF NOT EXISTS decision_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER,
    user_name       TEXT,
    timestamp       TIMESTAMP,
    event_type      TEXT,      -- e.g. 'decision_made', 'decision_changed', 'review_completed'
    choice          TEXT,      -- e.g. 'home', 'draw', 'away'
    confidence      INTEGER,   -- 1-5 或 0-100，由 UI 定义
    reason_tags     TEXT,      -- 逗号分隔的标签，如 'injury,form,odds'
    reason_detail   TEXT,      -- 自由文本理由
    changed_from    TEXT,      -- 若是修改决策，记录原选择
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

-- 5. 比赛结果表：用于赛后复盘对照
CREATE TABLE IF NOT EXISTS results (
    match_id     INTEGER PRIMARY KEY,
    home_goals   INTEGER,
    away_goals   INTEGER,
    result       TEXT,       -- 'H' / 'D' / 'A'
    is_finished  BOOLEAN DEFAULT 0,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);
"""


def init_database(db_path: str = DB_PATH) -> None:
    """创建数据库文件及全部 5 张表（若不存在）。"""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        print(f"数据库已初始化：{db_path}")

        # 打印已创建的表，便于核对
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"共 {len(tables)} 张表：{tables}")
    finally:
        conn.close()


if __name__ == "__main__":
    init_database()
