import pandas as pd
import os

df = pd.read_parquet("data/RMDC26_Beginner_Tier_test.parquet")

for filter_name in df['filt'].unique():
    df_filter = df[df['filt'] == filter_name]
    for name, group in df_filter.groupby("name"):
        os.makedirs(f"data/data_{filter_name}", exist_ok=True)
        group.to_csv(f"data/data_{filter_name}/{name}.csv", columns=['bjd', 'mag', 'mag_err'], index=False, header=False)
names = df['name'].unique()
Ra = []
Dec = []
for name in names:
    df_name = df[df['name'] == name]
    Ra.append(df_name['ra_deg'].iloc[0])
    Dec.append(df_name['dec_deg'].iloc[0])
out_df = pd.DataFrame({'name': names, 'ra_deg': Ra, 'dec_deg': Dec})
out_df.to_csv("data/coords.csv", index=False)