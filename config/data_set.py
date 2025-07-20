# 導入自定義的快取與名稱紀錄檔路徑
from config.paths import cachepath, cachepath_namelog

# 導入自定義的基本隨機編碼產生函式
from StevenTricks.code_utils import basic_code

# 導入自定義的檔案整理函式（含目錄走訪與欄位解析）
from StevenTricks.file_utils import PathWalk_df

# 導入自定義的 pickle 存取工具
from StevenTricks.file_utils import pickleload, picklesave

# 標準函式庫，用來處理目錄清空與重新建立
import shutil
from pathlib import Path


# 🔹 要建立快取記錄的資料項目（會對每一個建立唯一 code）
data_list = ["個別產業成交比重", "產業及三大法人", "產業及融資融券"]


def update_cache(data_list, cache_dir, namelog_path, code_length=4, avoid_mode="fuzzy"):
    """
    根據 data_list 和現有的 cachename_log 進行同步更新：
    - 清除失效的 code（在 cache 資料夾中找不到的）
    - 新增 data_list 中的新項目
    - 保留有效的 code

    參數說明：
    - data_list: 使用者關心的資料標籤列表
    - cache_dir: 快取資料儲存的資料夾路徑
    - namelog_path: 用來儲存 log.pkl 的路徑，內容是每筆資料對應的 code
    - code_length: 每個 code 預設長度（隨機編碼 + 數字編碼）
    - avoid_mode: 避免重複的比對方式，"fuzzy"（模糊比對）或 "exact"（完全相同）
    """

    # 將傳入的路徑字串轉為 Path 物件以利後續操作
    cache_dir = Path(cache_dir)
    namelog_path = Path(namelog_path)

    # 🔸 取得目前 cache 目錄下所有 .pkl 檔案的 code 前綴（透過檔名前段）
    cache_files = [f.stem.split("_")[0] for f in cache_dir.glob("*.pkl")]

    # 🔸 如果 namelog 已存在，代表已有歷史對應記錄，要比對與更新
    if namelog_path.exists():
        log_df = {}  # 最終更新的對應表
        log_df_old = pickleload(namelog_path)  # 讀入歷史記錄

        # 🔸 使用 PathWalk_df 取得所有快取檔案對應欄位（包含 code 欄）
        cache_walk = PathWalk_df(
            cache_dir,
            fileinclude=[".pkl"],
            name_format="code_time_order.ext"  # 命名格式協助解析出 code 欄
        )

        # 🔸 抓出目前 cache 裡面所有有效的 code 列表（避免重複使用）
        avoid_list = cache_walk["code"].unique().tolist()

        # 🔸 對每一筆歷史記錄檢查是否還有效
        for key, code in log_df_old.items():
            if code in avoid_list:
                # ✅ 此 code 在 cache 中仍然有效，保留
                log_df[key] = code
            else:
                # ❌ 此 code 已失效，需刪除其對應檔案（若殘留）並重建新的 code
                for filepath in cache_walk[cache_walk["code"] == code]["path"]:
                    p = Path(filepath)
                    if p.exists():
                        p.unlink()  # 刪除該檔案

                # 🔸 使用 basic_code 建立新的唯一 code，並避免與現有重複
                code = basic_code(length=code_length, match_mode=avoid_mode, avoid_list=[code])
                log_df[key] = code

    else:
        # 🔸 若 namelog 不存在，代表首次建立，需清空資料夾重建快取目錄
        shutil.rmtree(cache_dir)  # 移除整個資料夾（若存在）
        cache_dir.mkdir(parents=True, exist_ok=True)  # 建立新的空資料夾

        # 🔸 為每個資料項目產生一組不重複 code
        code = basic_code(length=code_length, match_mode=avoid_mode, count=len(data_list))

        # 🔸 建立名稱與 code 的對應字典
        log_df = dict(zip(data_list, code))

    # 🔸 將更新後的 code 對應表儲存為 pickle
    picklesave(log_df, namelog_path)

    # ✅ 回傳最後的對應表結果
    return log_df


# 🔹 若直接執行此模組，將執行更新流程並印出結果
if __name__ == "__main__":
    cache_data = update_cache(data_list, cachepath, cachepath_namelog, 5, "fuzzy")
