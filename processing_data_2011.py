import pandas as pd
import glob
import os

def process_by_month(input_path, output_path):
    df = pd.read_parquet(input_path)
    df.columns = [c.lower() for c in df.columns]

    df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0)]
    
    df['time_bin'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.floor('30min')

    summary = df.groupby(['time_bin', 'pulocationid']).size().reset_index(name='demand_count')
    
    summary.to_csv(output_path, index=False)

files = glob.glob(r'/content/drive/MyDrive/Mining of massive dataset/data/2011/*.parquet')
for f in files:
    out_name = f.replace('.parquet', '_summary.csv')
    process_by_month(f, out_name)