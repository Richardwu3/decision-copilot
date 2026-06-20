"""
database/db.py
================
Google Sheets 数据层，用于多用户决策日志存储（取代 SQLite 的 decision_events 表）。

设计原则：
- 只负责"决策日志"的读写（用户做出的选择、信心、理由，以及当时AI的判断快照）。
  比赛/预测/特征/结果数据仍由 worldcup_predictor_v4.py 和 schedule_2026_result.xls
  提供，本文件不重复实现那部分逻辑。
- MVP 阶段预测不可修改，只追加新记录（即没有 update_decision，只有 save_decision 追加一行）。
- Google Sheets 不可用时（凭证缺失、网络问题、API 配额等）不能让主应用崩溃，
  所有公开函数在失败时返回空结果并打印警告，调用方（app.py）据此降级展示。

表头字段（与产品设计保持一致，对应原 SQLite decision_events 表 + AI 快照两个新增字段）：
    timestamp       决策写入时间（ISO格式字符串）
    user_name       用户昵称（取代原来的 session_id）
    match_id        比赛ID
    choice          用户选择：'home' / 'draw' / 'away'
    confidence      用户信心，1-5
    reason          用户决策理由（自由文本）
    ai_choice       写入决策时刻，AI 的预测结果：'home' / 'draw' / 'away'
    ai_confidence   写入决策时刻，AI 对该预测的置信度（0-1之间的概率，取AI预测结果对应的概率值）

依赖：gspread, google-auth (oauth2client 已被 Google 官方弃用，改用 google-auth + google-auth-oauthlib)
"""

import os
import json
from datetime import datetime

# gspread / google-auth 是可选依赖：如果没装或没配置凭证，
# 整个模块仍应能被 import，只是所有函数会优雅地返回空值。
try:
    import gspread
    from google.oauth2.service_account import Credentials
    _GSPREAD_AVAILABLE = True
except ImportError:
    _GSPREAD_AVAILABLE = False


CREDENTIALS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "credentials.json"
)

# Google Sheet 名称与工作表（tab）名称。
# 如果你的 Sheet 名称不同，改这两个常量即可，不需要改下面的逻辑。
SPREADSHEET_NAME = "Decision_Logs"
WORKSHEET_NAME = "Sheet1"

SHEET_HEADERS = [
    "timestamp", "user_name", "match_id", "choice", "confidence",
    "reason", "ai_choice", "ai_confidence"
]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]


# ============================================================
# 内部：连接管理
# ============================================================

_worksheet_cache = None  # 模块级缓存，避免每次调用都重新认证


def _get_worksheet():
    """
    返回已打开的 worksheet 对象；任何环节失败都返回 None 并打印警告，
    不抛出异常（保证主应用在 Sheets 不可用时仍能运行，只是决策记录功能降级）。
    """
    global _worksheet_cache
    if _worksheet_cache is not None:
        return _worksheet_cache

    if not _GSPREAD_AVAILABLE:
        print("[db.py] 警告：未安装 gspread / google-auth，决策日志功能不可用。"
              "请运行 pip install gspread google-auth")
        return None

    if not os.path.exists(CREDENTIALS_PATH):
        print(f"[db.py] 警告：未找到凭证文件 {CREDENTIALS_PATH}，决策日志功能不可用。")
        return None

    try:
        client = gspread.service_account(filename=CREDENTIALS_PATH)
    except Exception as e:
        print(f"[db.py] 警告：Google Sheets 认证失败（{e}），决策日志功能不可用。")
        return None

    try:
        spreadsheet = client.open(SPREADSHEET_NAME)
    except Exception as e:
        # 临时修改：打印完整的异常类型和详细信息
        print(f"[db.py] 警告：无法打开 Google Sheet '{SPREADSHEET_NAME}'")
        print(f"  异常类型：{type(e).__name__}")
        print(f"  异常详情：{e}")
        print(f"  请确认表格名称正确，且已与 service account 邮箱共享编辑权限。")
        return None

    try:
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        # 工作表不存在则自动创建并写入表头，降低部署门槛
        try:
            worksheet = spreadsheet.add_worksheet(
                title=WORKSHEET_NAME, rows=1000, cols=len(SHEET_HEADERS)
            )
            worksheet.append_row(SHEET_HEADERS)
            print(f"[db.py] 已自动创建工作表 '{WORKSHEET_NAME}' 并写入表头。")
        except Exception as e:
            print(f"[db.py] 警告：创建工作表失败（{e}），决策日志功能不可用。")
            return None
    except Exception as e:
        print(f"[db.py] 警告：打开工作表失败（{e}），决策日志功能不可用。")
        return None

    # 确保表头存在且与预期一致（首行为空时补写表头）
    try:
        existing_header = worksheet.row_values(1)
        if not existing_header:
            worksheet.append_row(SHEET_HEADERS)
    except Exception as e:
        print(f"[db.py] 警告：校验表头失败（{e}），继续尝试使用现有工作表。")

    _worksheet_cache = worksheet
    return worksheet


def _safe_get_all_records():
    """
    包装 worksheet.get_all_records()，处理空表、表头不一致等问题。
    返回 list[dict]，失败时返回空列表。
    """
    ws = _get_worksheet()
    if ws is None:
        return []

    try:
        records = ws.get_all_records()
        return records
    except Exception as e:
        print(f"[db.py] 警告：读取 Google Sheets 数据失败（{e}），返回空记录。")
        return []


# ============================================================
# 公开函数 1：get_user_history
# ============================================================

def get_user_history(user_name: str):
    """
    读取该用户的所有决策记录，按 timestamp 升序排列（最早的在前，
    便于在 UI 上按时间线展示）。

    返回：list[dict]，每个 dict 的 key 为 SHEET_HEADERS 中的字段名。
         Google Sheets 不可用或用户无记录时返回空列表 []。
    """
    if not user_name:
        return []

    all_records = _safe_get_all_records()
    user_records = [r for r in all_records if str(r.get("user_name", "")) == str(user_name)]

    try:
        user_records.sort(key=lambda r: r.get("timestamp", ""))
    except Exception:
        pass  # 排序失败不影响返回结果，只是顺序可能不严格

    return user_records


# ============================================================
# 公开函数 2：save_decision
# ============================================================

def save_decision(user_name: str,
                  match_id,
                  choice: str,
                  confidence,
                  reason: str = "",
                  ai_choice: str = "",
                  ai_confidence=None) -> bool:
    """
    追加一条新的决策记录到 Google Sheets。
    MVP 阶段不支持修改/覆盖已有记录——每次调用都是新增一行，
    即使同一用户对同一场比赛重复提交，也会保留多条历史（取最新一条作为当前决策，
    由调用方在读取时自行处理，本函数只负责追加）。

    参数：
        user_name       用户昵称
        match_id        比赛ID（int 或 str 均可，写入时转为字符串）
        choice          用户选择：'home' / 'draw' / 'away'
        confidence      用户信心，1-5
        reason          决策理由（可选，默认空字符串）
        ai_choice       当时AI预测的结果，'home'/'draw'/'away'（可选）
        ai_confidence   当时AI对该预测结果的置信度，0-1之间的浮点数（可选）

    返回：True 表示写入成功，False 表示 Google Sheets 不可用或写入失败
         （此时不影响调用方继续运行，仅决策记录不会被持久化）。
    """
    ws = _get_worksheet()
    if ws is None:
        print("[db.py] 警告：Google Sheets 不可用，本次决策未被持久化。")
        return False

    row = [
        datetime.now().isoformat(),
        str(user_name),
        str(match_id),
        str(choice),
        str(confidence) if confidence is not None else "",
        str(reason) if reason else "",
        str(ai_choice) if ai_choice else "",
        f"{ai_confidence:.4f}" if isinstance(ai_confidence, (int, float)) else "",
    ]

    try:
        ws.append_row(row, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        print(f"[db.py] 警告：写入 Google Sheets 失败（{e}），本次决策未被持久化。")
        return False


# ============================================================
# 公开函数 3：get_all_decisions
# ============================================================

def get_all_decisions():
    """
    读取全部用户的全部决策记录（用于 Trust Analytics：人机一致率、
    冲突时谁更准等需要跨用户聚合统计的场景）。

    返回：list[dict]，Google Sheets 不可用时返回空列表 []。
    """
    return _safe_get_all_records()


# ============================================================
# 自检（直接运行本文件可验证连接是否正常，不影响 app.py 的 import）
# ============================================================

if __name__ == "__main__":
    print("正在检测 Google Sheets 连接...")
    ws = _get_worksheet()
    if ws is None:
        print("连接失败，请检查 credentials.json 是否存在、"
              "Sheet 名称是否正确、是否已与 service account 邮箱共享编辑权限。")
    else:
        print(f"连接成功：已打开工作表 '{WORKSHEET_NAME}'。")
        all_records = get_all_decisions()
        print(f"当前共有 {len(all_records)} 条决策记录。")
