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

# streamlit 同样按可选依赖处理（任务1：用于检测 st.secrets 是否包含
# Google Service Account 配置）。db.py 在被 app.py import 时 streamlit
# 必然可用；但本文件也支持被独立脚本调用（见文末 if __name__ == "__main__"
# 自检块），那种场景下不强制要求 streamlit 已安装。
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


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


def _load_credentials_from_secrets():
    """
    任务1：检测 st.secrets 中是否存在 Google Service Account 配置。

    优先查找的 key 名为 "gcp_service_account"（Streamlit 官方文档推荐的
    标准约定，参见 docs.streamlit.io 的 Google Sheets 连接教程）。
    如果你的 secrets.toml / Streamlit Cloud Secrets 中使用了不同的 key 名，
    请同步修改下面的 _SECRETS_KEY 常量，不需要改动其他逻辑。

    返回：
        dict —— 找到凭证配置时，返回可直接 json.dump 的凭证字典
        None —— 本地开发环境（无 st.secrets 或其中没有该 key）时返回 None，
                调用方据此回退到本地 credentials.json，不影响本地行为。
    """
    if not _STREAMLIT_AVAILABLE:
        return None

    _SECRETS_KEY = "gcp_service_account"

    try:
        # st.secrets 在本地没有 .streamlit/secrets.toml 时访问会抛异常，
        # 用 try/except 而不是 "in" 判断，兼容所有 Streamlit 版本的行为差异。
        if _SECRETS_KEY not in st.secrets:
            return None
        secrets_section = st.secrets[_SECRETS_KEY]
    except Exception:
        # 本地未配置 st.secrets 是完全正常的情况（本地开发场景），
        # 静默返回 None，不打印警告，避免本地开发时产生误导性输出。
        return None

    try:
        # st.secrets 返回的是 AttrDict，转成普通 dict 才能被 json.dump 序列化
        return dict(secrets_section)
    except Exception as e:
        print(f"[db.py] 警告：st.secrets['{_SECRETS_KEY}'] 存在但格式无法解析（{e}），"
              "尝试回退到本地 credentials.json。")
        return None


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

    # ---- 任务1：本地 credentials.json 与 Streamlit Cloud st.secrets 兼容 ----
    # 优先级：st.secrets 中存在 Google Service Account 配置 -> 用它（云端场景）；
    #        否则退回读取本地 credentials.json（本地开发场景，行为完全不变）。
    # 两条路径最终都调用同一个 gspread.service_account(filename=...)，
    # 区别只在于这个 filename 指向的是本地原文件还是临时生成的文件。
    client = None
    tmp_credentials_path = None

    secrets_dict = _load_credentials_from_secrets()
    if secrets_dict is not None:
        try:
            import tempfile
            # 用 tempfile 创建临时凭证文件，写入后立即用于认证，
            # finally 块确保 _get_worksheet 返回前一定清理掉这个临时文件，
            # 不在磁盘上长期保留 Service Account 私钥的明文副本。
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as tmp_file:
                json.dump(secrets_dict, tmp_file)
                tmp_credentials_path = tmp_file.name

            client = gspread.service_account(filename=tmp_credentials_path)
        except Exception as e:
            print(f"[db.py] 警告：从 st.secrets 读取 Google 凭证失败（{e}），"
                  "尝试回退到本地 credentials.json。")
            client = None
        finally:
            if tmp_credentials_path is not None and os.path.exists(tmp_credentials_path):
                try:
                    os.remove(tmp_credentials_path)
                except Exception:
                    pass  # 清理失败不影响主流程，临时文件位于系统临时目录，不影响功能

    if client is None:
        # 本地开发路径：与修改前完全一致的行为
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
