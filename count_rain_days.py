import pandas as pd

def count_rain_stats(file_path):
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Chuyển cột time sang datetime và set làm index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # --- 1. Thống kê theo mốc 30 phút ---
    count_30min_none = df['Rain_None'].sum()
    count_30min_light = df['Rain_Light'].sum()
    count_30min_moderate = df['Rain_Moderate'].sum()
    count_30min_heavy = df['Rain_Heavy'].sum()
    
    print("=== THỐNG KÊ THEO MỐC 30 PHÚT ===")
    print(f"Số mốc không mưa: {count_30min_none}")
    print(f"Số mốc mưa nhỏ  : {count_30min_light}")
    print(f"Số mốc mưa vừa  : {count_30min_moderate}")
    print(f"Số mốc mưa to   : {count_30min_heavy}\n")
    
    # --- 2. Thống kê theo NGÀY ---
    # Gom nhóm theo ngày ('D') và tính tổng số mốc trong ngày đó
    daily_df = df[['Rain_None', 'Rain_Light', 'Rain_Moderate', 'Rain_Heavy']].resample('D').sum()
    
    # Đếm số ngày: Điều kiện > 0 nghĩa là ngày đó có ít nhất 1 mốc 30 phút xuất hiện loại mưa đó
    days_with_light = (daily_df['Rain_Light'] > 0).sum()
    days_with_moderate = (daily_df['Rain_Moderate'] > 0).sum()
    days_with_heavy = (daily_df['Rain_Heavy'] > 0).sum()
    
    # Ngày hoàn toàn không mưa là ngày mà cả 48 mốc (24h * 2) đều là Rain_None
    days_completely_dry = (daily_df['Rain_None'] == 48).sum()
    
    print("=== THỐNG KÊ THEO NGÀY ===")
    print(f"Số ngày hoàn toàn không mưa : {days_completely_dry} ngày")
    print(f"Số ngày có xuất hiện mưa nhỏ: {days_with_light} ngày")
    print(f"Số ngày có xuất hiện mưa vừa: {days_with_moderate} ngày")
    print(f"Số ngày có xuất hiện mưa to : {days_with_heavy} ngày")
    
    # Lưu ý: Tổng số ngày của các loại mưa có thể lớn hơn 365/366 
    # vì một ngày có thể vừa có lúc mưa nhỏ, vừa có lúc mưa to.

if __name__ == "__main__":
    # Thay bằng đường dẫn tới file weather_2024.csv hoặc weather_2018.csv của bạn
    file_path = r'D:\StudyStuff\Mining Massive Dataset\mining_of_massive_dataset\weather_2024.csv'
    count_rain_stats(file_path)