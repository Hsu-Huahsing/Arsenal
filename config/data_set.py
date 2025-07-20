from config.paths import cachepath,cachepath_namelog
from StevenTricks.code_utils import basic_code
from StevenTricks.file_utils import PathWalk_df
from StevenTricks.file_utils import pickleload,picklesave
import shutil
from pathlib import Path


data_list = ["個別產業成交比重","產業及三大法人","產業及融資融券"]



def update_cachename_log(data_list, cache_dir, namelog_path, code_length=4, avoid_mode="fuzzy"):
    """
    根據 data_list 和現有的 cachename_log 進行同步更新：
    - 清除失效的 code（在 cache 資料夾中找不到的）
    - 新增 data_list 中的新項目
    - 保留有效的 code
    """

    cache_dir = Path(cache_dir)
    namelog_path = Path(namelog_path)

    # 🔹 讀取 cache 裡面現有的檔案名稱（stem）
    cache_files = [f.stem.split("_")[0] for f in cache_dir.glob("*.pkl")]

    # 🔹 如果有 namelog 檔案，讀進來
    if namelog_path.exists():
        log_df = {_:None for _ in data_list}
        log_df_old = pickleload(namelog_path)
        cache_walk = PathWalk_df(cache_dir,fileinclude=[".pkl"],name_format="code_time_order.ext")
        avoid_list = cache_walk["code"].unique().tolist()
        for key,code in log_df_old.items():
            if code in avoid_list:
                log_df[key] = code
            elif code not in avoid_list:
                for filepath in cache_walk[cache_walk["code"] == code]["filepath"]:
                    p = Path(filepath)
                    if p.exists():
                        p.unlink()
                code = basic_code(length=code_length, match_mode=avoid_mode,avoid_list=[code])
                log_df[key] = code
    else:
        shutil.rmtree(cache_dir)  # 整個目錄刪除
        cache_dir.mkdir(parents=True, exist_ok=True)  # 重建空目錄
        code = basic_code(length=code_length, match_mode=avoid_mode,count=len(data_list))
        log_df = dict(zip(data_list,code ))

    picklesave(log_df,namelog_path)
    return log_df

