"""
init_db.py
==========
初始化 Decision Review Copilot 的 SQLite 数据库（Phase 1，5 张表）。

运行方式：
    python database/init_db.py                  # 仅初始化（CREATE TABLE IF NOT EXISTS）
    python database/init_db.py --check           # 校验并自动修复 schema（见 ensure_database_schema）
    python database/init_db.py --debug           # 打印当前数据库完整 schema（见 print_database_schema）

幂等：重复运行不会破坏已有数据（CREATE TABLE IF NOT EXISTS）。
若需要完全重建，删除 decision_copilot.db 文件后重新运行即可。

----------------------------------------------------------------
背景（Streamlit Cloud 部署排查记录）：
之前 app.py 仅在 get_connection() 里调用 sqlite3.connect(db_path)，
而 sqlite3.connect() 在数据库文件不存在时会静默创建一个 0 张表的
空文件——不会报错，也不会自动建表。本地开发环境因为开发者手动
运行过一次本脚本，数据库文件早已存在且 schema 完整，问题被掩盖；
但 Streamlit Cloud 的每次全新部署都是一个干净容器，从未有人在
其中手动跑过 init_db.py，导致首次访问 Dashboard 时对一个空数据库
执行 SELECT，触发 sqlite3.OperationalError: no such table: matches。

修复方式：app.py 启动时改为调用 database.db_utils.ensure_database_schema()，
该函数不依赖"数据库文件是否存在"，而是直接检查"所有期望的表和列
是否真的存在"，缺失则自动创建/重建，因此本地和 Streamlit Cloud
都能保证拿到一个 schema 完整可用的数据库连接，不再需要任何手动
预先执行的步骤。
----------------------------------------------------------------
"""

import sqlite3
import os
import sys

# 数据库文件路径：与本脚本同级目录下的上一级（项目根目录）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_PROJECT_ROOT, "decision_copilot.db")

# 确保 "python database/init_db.py"（直接按文件路径运行脚本）这种调用方式
# 也能正确解析 "from database.db_utils import ..." 这类绝对导入。
# 原因：直接用文件路径运行脚本时，Python 会把 sys.path[0] 设为脚本所在目录
# （即 database/ 目录本身），而不是项目根目录，导致 Python 在 sys.path 里
# 找不到 "database" 这个包名自己（database/ 目录的父目录才是 database 包的
# 容器）。用 "python -m database.init_db" 方式运行则不受影响，因为 -m 会把
# 当前工作目录加入 sys.path。为了让两种运行方式都能正常工作（文档里一直
# 建议的是更直观的 "python database/init_db.py" 写法），这里显式把项目根
# 目录插入 sys.path，且只在它还不存在时插入，避免重复。
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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
    if "--check" in sys.argv:
        # 延迟导入，避免 init_db.py 在仅做基础初始化时引入不必要的依赖循环
        from database.db_utils import get_connection, ensure_database_schema
        conn = get_connection()
        conn = ensure_database_schema(conn)
        conn.close()
    elif "--debug" in sys.argv:
        from database.db_utils import get_connection, print_database_schema
        conn = get_connection()
        print_database_schema(conn)
        conn.close()
    else:
        init_database()
