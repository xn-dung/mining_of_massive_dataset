import pandas as pd
import os
import gc
import re

def process_by_month_optimized(input_path, base_output_dir='/content/drive/MyDrive/Mining of massive dataset/preprocessing_output'):
    basename = os.path.basename(input_path)

    # Trích xuất Năm (YYYY) và Tháng (MM) từ tên file
    match = re.search(r'(20\d{2})-(\d{2})', basename)
    if match:
        year = match.group(1)
        month = match.group(2)
    else:
        year = "Unknown"
        month = basename.replace('.parquet', '')

    print(f"Đang xử lý: Năm {year} - Tháng {month}...")

    # Tạo Folder theo định dạng "yellow_năm"
    year_folder_name = f"yellow_{year}"
    output_dir = os.path.join(base_output_dir, year_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Định nghĩa trước tên file đầu ra để kiểm tra
    vol_out_path = os.path.join(output_dir, f"{month}_volume.csv")
    flow_out_path = os.path.join(output_dir, f"{month}_flow.parquet")

    if os.path.exists(vol_out_path) and os.path.exists(flow_out_path):
        print(f"Bỏ qua (Đã tồn tại): {month}_volume.csv & {month}_flow.parquet")
        return

    # --- ĐỌC VÀ LỌC DỮ LIỆU ---
    try:
        df = pd.read_parquet(input_path, columns=[
            'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'trip_distance', 'fare_amount', 'passenger_count',
            'PULocationID', 'DOLocationID'
        ])
    except ValueError:
        df = pd.read_parquet(input_path, columns=[
            'pickup_datetime', 'dropoff_datetime',
            'trip_distance', 'fare_amount', 'passenger_count',
            'PULocationID', 'DOLocationID'
        ])

    df.columns = [c.lower() for c in df.columns]
    pickup_col = 'tpep_pickup_datetime' if 'tpep_pickup_datetime' in df.columns else 'pickup_datetime'
    dropoff_col = 'tpep_dropoff_datetime' if 'tpep_dropoff_datetime' in df.columns else 'dropoff_datetime'

    # Lọc các chuyến đi lỗi
    df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0)]
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 7)]
    df = df.drop(columns=['trip_distance', 'fare_amount', 'passenger_count'])

    df[pickup_col] = pd.to_datetime(df[pickup_col])
    df[dropoff_col] = pd.to_datetime(df[dropoff_col])
    travel_time_sec = (df[dropoff_col] - df[pickup_col]).dt.total_seconds()
    df = df[travel_time_sec > 0]
    df = df.drop(columns=[dropoff_col])

    # Lọc Bounding Box NYC (LocationID: 1 -> 263)
    df = df[(df['pulocationid'] >= 1) & (df['pulocationid'] <= 263)]
    df = df[(df['dolocationid'] >= 1) & (df['dolocationid'] <= 263)]

    # --- TẠO TIME BIN ---
    df['time_bin'] = df[pickup_col].dt.floor('30min')
    df = df.drop(columns=[pickup_col])

    # ==============================================================
    # TÁCH 2 LUỒNG VÀ LƯU FILE THEO THÁNG
    # ==============================================================

    start_vol = df.groupby(['time_bin', 'pulocationid']).size().reset_index(name='start_volume')
    start_vol.rename(columns={'pulocationid': 'locationid'}, inplace=True)

    end_vol = df.groupby(['time_bin', 'dolocationid']).size().reset_index(name='end_volume')
    end_vol.rename(columns={'dolocationid': 'locationid'}, inplace=True)

    volume_df = pd.merge(start_vol, end_vol, on=['time_bin', 'locationid'], how='outer').fillna(0)
    
    volume_df['start_volume'] = volume_df['start_volume'].astype(int)
    volume_df['end_volume'] = volume_df['end_volume'].astype(int)

    volume_df.to_csv(vol_out_path, index=False)

    del start_vol, end_vol, volume_df

    flow_df = df.groupby(['time_bin', 'pulocationid', 'dolocationid']).size().reset_index(name='flow_count')

    flow_df.to_parquet(flow_out_path, engine='pyarrow', compression='snappy')

    print(f"Đã lưu vào {year_folder_name}/ (File: {month}_volume.csv & {month}_flow.parquet)")

    del df, flow_df
    gc.collect()