from conf import dbpath_cleaned,numericol
from StevenTricks.dbsqlite import readsql_iter
from StevenTricks.convert_utils import safe_numeric_convert
import pandas as pd


def diff_previous(df: pd.DataFrame, sort_col : ["代號","date"], item: str,subitem:str):
    temp = df.copy()
    temp = temp.sort_values(by=sort_col, ascending=False)
    value_col = numericol[item][subitem]
    value_col = [_ for _ in value_col if _ in temp]
    temp = safe_numeric_convert(temp, value_col)
    # 對 value_col 欄位，根據 "代號" 分組後做 shift
    for col in value_col:
        temp[f"previous_{col}"] = temp.groupby("代號")[col].shift(-1)
        temp[f'diff_{col}'] = temp[col] - temp[f"previous_{col}"]
    return temp

if __name__ == "__main__":
    table1 = readsql_iter(dbpath=dbpath_cleaned,db_list=["外資及陸資投資持股統計.db"])
    table1 = next(table1)

