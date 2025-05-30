
import pandas as pd
from conf import colname_dic, dropcol, numericol, datecol, key_set
from StevenTricks.convert_utils import changetype_stringtodate
from StevenTricks.dictur import keyinstr


def key_extract(dic):
    dict_df_list = []
    for step,set_i in key_set.items():
        dict_df = {}
        print(step,set_i)
        if step in ["main1","main2"]:
            print("main")
            cnt = 0
            while True:
                print(cnt)
                for key,item in set_i.items():
                    print(key,item)
                    for col in dic:
                        print(col)
                        if cnt == 0 and item == col:
                            print(item)
                            dict_df[key] = dic[col]
                            break
                        elif cnt != 0 and item+str(cnt) == col:
                            print(item+str(cnt))
                            dict_df[key] = dic[col]
                            break
                print(dict_df)
                if dict_df :
                    print(dict_df)
                    dict_df_list.append(dict_df)
                    dict_df = {}
                elif cnt >1 and not dict_df :
                    print(cnt,dict_df)
                    break
                cnt += 1
        else:
            print(step)
            for key, item in set_i.items():
                print(key,item)
                for col in dic:
                    print(col)
                    if item == col:
                        dict_df[key] = dic[col]
            if dict_df:
                dict_df_list.append(dict_df)
    return dict_df_list


def rename_columns_batch(df, replace_pairs):
    """依照替換規則（old, new）依序套用第一個匹配欄位"""
    colstr = ",".join(df.columns)
    for old, new in replace_pairs:
        colstr = colstr.replace(old, new, 1)
    df.columns = colstr.split(",")
    return df


def safe_numeric_convert(df, cols):
    """將指定欄位轉成數值型態，無法轉換設為 NaN"""
    cols = [_ for _ in cols if _ in df]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df


def frameup_safe(data_dict):
    # 如果沒有groups就直接用這個，和groups會返回一樣的東西
    df = pd.DataFrame(data_dict["data"])
    col_diff = list(range(0, df.shape[1] - len(data_dict["fields"])))
    col = data_dict["fields"] + col_diff
    col = [colname_dic.get(_,_) for _ in col]
    df.columns = col
    data_dict["data_cleaned"] = df.drop(columns=col_diff)
    return data_dict


def data_cleaned_df(df, item, subitem, date):
    df = df.replace({",": "", r'\)': '', r'\(': '_'}, regex=True)
    df = df.drop(columns=dropcol, errors='ignore')
    cols = numericol[item][subitem]
    df = safe_numeric_convert(df, cols)
    date_col = [_ for _ in datecol if _ in df]
    if date_col:
        df = changetype_stringtodate(df, date_col, mode=3)
        # df = df.set_index(datecol, drop=True)
    else :
        df["date"] = [date]*df.shape[0]
    return df

def data_cleaned_groups(data_dict):
    df_dict = {}
    df_main = pd.DataFrame(data_dict["data"])
    df_col = []
    cnts = 0
    for dict_temp in data_dict["groups"]:
        cnts += 1
        if not df_col:
            df_col = list(range(0, dict_temp["start"]))
        if cnts == len(data_dict["groups"]):
            df_temp = pd.DataFrame(df_main.iloc[:, df_col + list(range(dict_temp["start"], len(data_dict["fields"])))])
        else:
            df_temp = pd.DataFrame(
                df_main.iloc[:, df_col + list(range(dict_temp["start"], dict_temp["start"] + dict_temp["span"]))])
        col = [data_dict["fields"][_] for _ in df_temp.columns]
        df_temp.columns = col
        df_dict[dict_temp["title"]] = df_temp
    data_dict["data_cleaned"] = df_dict
    return data_dict