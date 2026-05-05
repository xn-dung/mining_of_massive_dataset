import pandas as pd
import glob
import os

DATA_DIR = '/content/drive/MyDrive/Mining of massive dataset/preprocessing_output/'
LOOK_UP_DIR = '/content/drive/MyDrive/Mining of massive dataset/preprocessing_output/taxi_zone_lookup_grid.csv'

def create_clean_master_dataset():
    print("1. Đọc và gộp toàn bộ các file Volume...")
    file_list = glob.glob('/content/drive/MyDrive/Mining of massive dataset/preprocessing_output/yellow_data/yellow_2018/' + '*_volume.csv')
    df_list = [pd.read_csv(file) for file in file_list]
    df_volume = pd.concat(df_list, ignore_index=True)

    if 'locationid' in df_volume.columns:
        df_volume = df_volume.rename(columns={'locationid' : 'LocationID'})

    df_volume['time_bin'] = pd.to_datetime(df_volume['time_bin'])

    # =========================================================================
    # BƯỚC LÀM SẠCH (DATA CLEANING) - XỬ LÝ TIME RÁC (JUMPING CLOCKS)
    # =========================================================================
    print(" -> Đang dọn dẹp các khung thời gian bị rác/nhảy đồng hồ...")

    # Bước 1.1: Chém đứt đuôi các năm lỗi, chỉ lấy đúng năm 2018
    df_volume = df_volume[df_volume['time_bin'].dt.year == 2018]

    # Bước 1.2: Ép thời gian về đúng vạch 30 phút (Time Snapping/Flooring)
    # Nếu đồng hồ nhảy ra '14:34:10', nó sẽ tự ép về '14:30:00'
    df_volume['time_bin'] = df_volume['time_bin'].dt.floor('30min')

    # Bước 1.3: Gộp (Sum) các dòng bị trùng lặp sau khi ép thời gian
    # Việc nhảy đồng hồ có thể tạo ra 2 dòng cho cùng vùng 43 lúc 00:30. Ta phải cộng dồn chúng lại.
    df_volume = df_volume.groupby(['LocationID', 'time_bin']).agg({
        'start_volume': 'sum',
        'end_volume': 'sum'
    }).reset_index()
    # =========================================================================

    print("2. Ghép với thông tin Lookup Grid...")
    df_lookup = pd.read_csv(LOOK_UP_DIR)
    df_volume = pd.merge(df_volume, df_lookup, on="LocationID", how="left")

    print("3. Đọc và chuẩn bị dữ liệu Ngữ cảnh (Holiday & Weather)...")
    df_holidays = pd.read_csv(DATA_DIR + 'holidays_2018.csv')
    df_weather = pd.read_csv(DATA_DIR + 'weather_2018.csv')

    df_context = pd.merge(df_holidays, df_weather, on="time", how="left")

    # Chuẩn hóa thời gian của Context
    df_context = df_context.rename(columns={'time': 'time_bin'})
    df_context['time_bin'] = pd.to_datetime(df_context['time_bin'])

    # Chắc chắn rằng file ngữ cảnh không có dòng bị trùng giờ (Duplicate)
    df_context = df_context.drop_duplicates(subset=['time_bin'], keep='first')

    print("4. Ghép toàn bộ thành 1 bảng duy nhất (Master Dataset)...")
    # QUAN TRỌNG: Dùng 'inner' join thay vì 'left' join!
    # File Context của bạn kéo dài đúng 1 năm 2018 (rất chuẩn).
    # Inner join sẽ tự động TỪ CHỐI bất kỳ dòng time rác nào của Volume nếu nó không khớp với lịch 2018.
    df_final = pd.merge(df_volume, df_context, on="time_bin", how="inner")

    df_final = df_final.sort_values(by=['time_bin', 'LocationID'])

    print("5. Đang lưu file Master Dataset Sạch...")
    output_parquet = DATA_DIR + 'master_dataset_2018_clean.parquet'
    output_csv = DATA_DIR + 'master_dataset_2018_clean.csv'

    df_final.to_parquet(output_parquet, index=False)
    # df_final.to_csv(output_csv, index=False)

    print(f"✅ Hoàn tất! Bảng dữ liệu đã SẠCH 100% không còn time rác.")
    print(f"Kích thước bảng dữ liệu: {df_final.shape[0]:,} dòng x {df_final.shape[1]} cột.")

    return df_final

if __name__ == '__main__':
    df_master_clean = create_clean_master_dataset()