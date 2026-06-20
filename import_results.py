"""
import_results.py
==================
一次性导入脚本：把 data/schedule_2026_result.xls 中已结束比赛的比分
写入 SQLite 的 results 表（任务2）。

背景：
  database/db_utils.py 中早就有 save_result() 函数，但项目里没有任何地方
  调用它，导致 results 表一直是空的，Match Detail 页面读不到真实比分。
  本脚本就是缺失的"调用方"。

数据来源列名（与 app.py 中 load_bookmaker_odds 读取同一份文件时使用的
列名保持一致，未改名）：
    match_id, home_goal, away_goal, result, home_xg, away_xg

判定"是否已结束"的规则：
    只处理 result 列不为空（非 NaN、非空字符串）的行。
    Excel 里 result 列具体编码成什么格式（W/D/L 还是 H/D/A 等）不重要——
    本脚本不读取、不信任这个字符串本身，只用它来判断"这场比赛是否已结束"。
    真正写入数据库的 H/D/A 由 save_result() 内部根据 home_goal/away_goal
    的大小关系重新计算，避免两边编码约定不一致导致的数据错误。

运行方式：
    python import_results.py

幂等性：
    save_result() 内部使用 INSERT OR REPLACE，按 match_id 主键覆盖写入，
    重复运行本脚本不会产生重复行，也不会报错。
"""

import os

import pandas as pd

from database.db_utils import get_connection, save_result

RESULT_FILE_PATH = "data/schedule_2026_result.xls"


def import_results(path: str = RESULT_FILE_PATH) -> int:
    """
    读取 Excel 中已结束比赛的比分，写入 SQLite results 表。
    返回成功导入的行数。
    """
    if not os.path.exists(path):
        print(f"[import_results] 错误：未找到文件 {path}，导入终止。")
        return 0

    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    required_cols = ["match_id", "home_goal", "away_goal", "result"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[import_results] 错误：文件缺少必需列 {missing}，导入终止。")
        return 0

    # 只处理 result 列不为空的行（表示该场比赛已结束）
    finished_df = df.dropna(subset=["result"]).copy()
    # 进一步过滤掉比分本身缺失的异常行（result有值但比分缺失，理论不应出现，但做防御性处理）
    finished_df = finished_df.dropna(subset=["home_goal", "away_goal"])

    print(f"[import_results] 文件中共 {len(df)} 场比赛，其中已结束 {len(finished_df)} 场。")

    conn = get_connection()
    imported = 0
    try:
        for _, row in finished_df.iterrows():
            match_id = int(row["match_id"])
            home_goals = int(row["home_goal"])
            away_goals = int(row["away_goal"])

            save_result(
                conn,
                match_id=match_id,
                home_goals=home_goals,
                away_goals=away_goals,
                is_finished=True,
            )
            imported += 1
            print(f"  match_id={match_id}: {home_goals} - {away_goals} 已写入")
    finally:
        conn.close()

    print(f"[import_results] 导入完成，共写入 {imported} 场比赛结果。")
    return imported


if __name__ == "__main__":
    import_results()
