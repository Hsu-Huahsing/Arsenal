from conf import dbpath_cleaned
from StevenTricks.dbsqlite import readsql_iter
import pandas as pd




def diff_previous(df: pd.DataFrame, group_col: str, value_col: str, new_col_prefix: str = "") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=[group_col, 'date'])
    df[f'{new_col_prefix}前日_{value_col}'] = df.groupby(group_col)[value_col].shift(1)
    df[f'{new_col_prefix}{value_col}_變化'] = df[value_col] - df[f'{new_col_prefix}前日_{value_col}']
    return df

if __name__ == "__main__":
    table1 = readsql_iter(dbpath=dbpath_cleaned,db_list=["外資及陸資投資持股統計.db"])
    table1 = next(table1)

    sort_col = ["代號","date"]
    value_col = ["外資及陸資尚可投資股數","全體外資及陸資持有股數","外資及陸資共用法令投資上限比率","陸資法令投資上限比率","發行股數","外資及陸資尚可投資比率","全體外資及陸資持股比率"]
    temp = table1.sort_values(by=sort_col, ascending=False)
    # 對 value_col 欄位，根據 "代號" 分組後做 shift
    temp[[f"前日_{col}" for col in value_col]] = (
        temp.groupby("代號")[value_col].shift(-1)
    )
    temp[value_col] = temp.groupby(sort_col)[value_col].shift(1)
    df[f'外資及陸資尚可投資股數_變化'] = df[value_col] - df[f'{new_col_prefix}前日_{value_col}']

    for a in table1:
        print(a)