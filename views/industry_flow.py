# 導入自定義的快取與名稱紀錄檔路徑
from config.paths import cachepath, cachepath_namelog
from StevenTricks.track_utils import cache_name

# 🔹 要建立快取記錄的資料項目（會對每一個建立唯一 code）
data_list = ["個別產業成交比重", "產業及三大法人", "產業及融資融券"]
data_dict = dict(zip(enumerate(data_list), data_list))
data = data_dict[0]

# 🔹 若直接執行此模組，將執行更新流程並印出結果
if __name__ == "__main__":
    if
    cache_data = cache_name(data_list, cachepath, cachepath_namelog, 5, "fuzzy")
