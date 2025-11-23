from pathlib import Path
import pandas as pd
from config.paths import dbpath_cleaned
from StevenTricks.internal_db import DBPkl

class TwseDB:
    """
    專門負責：從 cleaned 資料夾讀 TWSE 類 DB（例如 三大法人買賣超日報）。
    把 DBPkl 的細節封裝起來。
    """
    def __init__(self, item: str, subitem: str):
        self.item = item
        self.subitem = subitem
        self.base_dir = Path(dbpath_cleaned) / item
        self.db = DBPkl(str(self.base_dir), subitem)

    def load_table(self, decode_links: bool = True) -> pd.DataFrame:
        df = self.db.load_db(decode_links=decode_links)
        return df

    def get_stock_df(
        self,
        stock_id: str,
        start=None,
        end=None,
    ) -> pd.DataFrame:
        df = self.load_table(decode_links=True)
        df = df[df["代號"] == stock_id].copy()
        df["date"] = pd.to_datetime(df["date"])
        if start is not None:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["date"] <= pd.to_datetime(end)]
        df = df.sort_values("date")
        return df
