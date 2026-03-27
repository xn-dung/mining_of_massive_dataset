import pandas as pd
import glob
import os

def process_tlc_data(file_path):
    # 1. Đọc file Parquet
    df = pd.read_parquet(file_path)
    
    # 2. Chuẩn hóa tên cột thành chữ thường
    df.columns = [col.lower() for col in df.columns]
    
    # 3. Chuyển đổi định dạng thời gian
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    # 4. Loại bỏ dữ liệu rỗng (Null) ở các cột quan trọng
    # PULocationID và DOLocationID là then chốt cho bài toán điểm nóng
    critical_cols = ['tpep_pickup_datetime', 'pulocationid', 'dolocationid', 'trip_distance', 'fare_amount']
    df = df.dropna(subset=critical_cols)
    
    # 5. Loại bỏ dữ liệu ngoại lệ (Outliers) theo nghiên cứu ST-NN [cite: 286]
    df = df[
        (df['passenger_count'] > 0) & 
        (df['trip_distance'] > 0) & 
        (df['fare_amount'] > 0) &
        (df['pulocationid'].notnull()) &
        (df['dolocationid'].notnull())
    ]
    
    # 6. Xử lý các cột phụ phí (điền 0 cho các giá trị NaN cũ)
    cols_to_fix = ['congestion_surcharge', 'airport_fee', 'improvement_surcharge']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 7. Trích xuất đặc trưng thời gian cho bài toán Demand [cite: 584, 743]
    # Phân nhóm theo 30 phút như mô hình DMVST-Net gợi ý [cite: 575]
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['time_bins'] = df['tpep_pickup_datetime'].dt.floor('30T') 
    
    return df

def aggregate_demand(df):
    """
    Hàm gom nhóm để tính toán Demand (Nhu cầu) theo khu vực và thời gian
    Phục vụ bài toán dự đoán số lượng xe cần điều động 
    """
    demand_df = df.groupby(['time_bins', 'pulocationid']).size().reset_index(name='demand_count')
    return demand_df

# Ví dụ áp dụng cho danh sách các file trong thư mục
all_files = glob.glob("data/*.parquet")
processed_data_list = []

for file in all_files:
    print(f"Đang xử lý: {file}")
    clean_df = process_tlc_data(file)
    
    # Bạn có thể lưu lại file đã sạch hoặc gom nhóm ngay để giảm dung lượng
    demand_summary = aggregate_demand(clean_df)
    processed_data_list.append(demand_summary)

# Kết hợp dữ liệu lớn từ 2011-2024
final_demand_df = pd.concat(processed_data_list, ignore_index=True)

# Lưu kết quả cuối cùng để huấn luyện mô hình
# final_demand_df.to_parquet("final_taxi_demand_2011_2024.parquet")