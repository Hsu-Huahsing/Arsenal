from data_cleaning.twse import process_twse_data

process_twse_data(
    use_local_db_staging=True,
    batch_size=10,
    bucket_mode="year",
)
