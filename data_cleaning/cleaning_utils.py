
import pandas as pd
from conf import colname_dic, dropcol, numericol, datecol, key_set, fields_span
from StevenTricks.convert_utils import changetype_stringtodate
from StevenTricks.dictur import keyinstr


def key_extract(dic):
    dict_df_list = []
    for step,set_i in key_set.items():
        dict_df = {}
        print(step,set_i)
        if step in ["main1"]:
            print("main")
            cnt = 0
            while True:
                print(cnt)
                for key,item_list in set_i.items():
                    print(key,item_list)
                    for col in dic:
                        print(col)
                        if cnt == 0 and col in item_list :
                            print(col)
                            dict_df[key] = dic[col]
                            break
                        elif cnt != 0 and col in [_+str(cnt) for _ in item_list]:
                            print(col)
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
            for key, item_list in set_i.items():
                print(key,item_list)
                for col in dic:
                    print(col)
                    if col in item_list:
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
    df_dict = fields_span[data_dict["subitem"]]
    df_main = pd.DataFrame(data_dict["data"])
    df_col = []

    for key in df_dict:
        if key in ["融資","融券"]:
            df_col += [key+_ for _ in data_dict["fields"][df_dict[key]["start"]:df_dict[key]["end"]]]
        else:
            df_col += data_dict["fields"][df_dict[key]["start"]:df_dict[key]["end"]]

    df_main.columns = df_col
    data_dict["data_cleaned"] = df_main
    return data_dict