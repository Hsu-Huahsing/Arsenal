# 導入自定義的快取與名稱紀錄檔路徑
from config.paths import cachepath, cachepath_namelog, cleaned_db_dict
from StevenTricks.track_utils import cache_name
from StevenTricks.internal_db import readsql_iter
# 🔹 要建立快取記錄的資料項目（會對每一個建立唯一 code）
data_list = ["個別產業成交比重", "產業及三大法人", "產業及融資融券"]
data_dict = {i: v for i, v in enumerate(data_list)}
data_name = data_dict[0]

# 🔹 若直接執行此模組，將執行更新流程並印出結果
if __name__ == "__main__":
    if data_name == "個別產業成交比重":
        base_df = readsql_iter(dbpath=cleaned_db_dict["三大法人買賣超日報"])
        cache_data = cache_name(data_list, cachepath, cachepath_namelog, 5, "fuzzy")
