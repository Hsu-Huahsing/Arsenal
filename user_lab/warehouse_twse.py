from user_lab.warehouse_twse import load_twse_dashboard

dash = load_twse_dashboard()
tw_overall_summary = dash["overall_summary"]
tw_orphan_pairs = dash["orphan"]

print(tw_overall_summary)
print(tw_orphan_pairs[["item", "subitem", "cleaned_path"]].head())
