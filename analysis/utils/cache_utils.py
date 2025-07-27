# 導入自定義的快取與名稱紀錄檔路徑
from config.paths import cachepath, cachepath_namelog
from config.data_set import data_list

from StevenTricks.track_utils import cache_name

# 🔹 若直接執行此模組，將執行更新流程並印出結果
if __name__ == "__main__":
    cache_data = cache_name(data_list, cachepath, cachepath_namelog, 5, "fuzzy")