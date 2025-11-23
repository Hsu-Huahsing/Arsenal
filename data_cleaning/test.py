from data_cleaning.twse import process_twse_data

process_twse_data(
    storage_mode="cloud_staging",
    batch_size=500,         # 每輪從雲端拉下來處理 500 個檔再同步回去
    bucket_mode="year",
)
"""
完全雲端（跟以前一樣，不 staging）：
process_twse_data(
    storage_mode="cloud",
    batch_size=None,        # 無所謂，cloud 模式用不到
    bucket_mode="year",     # 例如每年一檔
)

雲端 + 本機暫存（你現在這個模式，只是換名稱）：
process_twse_data(
    storage_mode="cloud_staging",
    batch_size=500,         # 每輪從雲端拉下來處理 500 個檔再同步回去
    bucket_mode="year",
)

完全本機模式（你說「沒有雲端路徑可以用」的情境）：
process_twse_data(
    storage_mode="local",
    batch_size=500,         # 想要分批也可以，不想管就 None
    bucket_mode="year",
)
"""