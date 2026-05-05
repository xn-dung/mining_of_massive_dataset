import pandas as pd
import numpy as np
import geopandas as gpd # Đừng quên import geopandas nhé

def analyze_rain_crisis_combined(master_data_path, shapefile_path, min_baseline=15):
    """
    Analyzes Traffic Perturbation for BUSY ZONES only.
    Combines 'Rain_Moderate' and 'Rain_Heavy' into 'Significant Rain'.
    Uses Shapefile (.shp) for precise Zone mapping.
    """
    print(f"1. Loading Final Master Dataset...")
    if master_data_path.endswith('.parquet'):
        df_master = pd.read_parquet(master_data_path)
    else:
        df_master = pd.read_csv(master_data_path)

    df_master['time_bin'] = pd.to_datetime(df_master['time_bin'])

    df_master['hour'] = df_master['time_bin'].dt.hour
    df_master['day_of_week'] = df_master['time_bin'].dt.dayofweek
    df_master['is_weekend'] = df_master['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    print("2. Calculating Intrinsic Demand (Baseline) from normal conditions...")
    normal_mask = (df_master['Rain_None'] == 1) & (df_master['Is_Legal_Holiday'] == 0)
    df_normal = df_master[normal_mask]

    baseline_df = df_normal.groupby(['LocationID', 'hour', 'is_weekend'])['start_volume'].mean().reset_index(name='baseline_volume')

    print("3. Calculating Perturbation (Delta Volume)...")
    df_analysis = pd.merge(df_master, baseline_df, on=['LocationID', 'hour', 'is_weekend'], how='inner')

    print(f" -> Filtering out quiet zones (Keeping baseline >= {min_baseline} trips)...")
    df_analysis = df_analysis[df_analysis['baseline_volume'] >= min_baseline]

    df_analysis['delta_volume'] = df_analysis['start_volume'] - df_analysis['baseline_volume']

    print("4. Classifying Light Rain vs. Moderate/Heavy Rain impact...")
    for col in ['Rain_Light', 'Rain_Moderate', 'Rain_Heavy']:
        if col in df_analysis.columns:
            df_analysis[col] = df_analysis[col].fillna(0).astype(int)

    df_light = df_analysis[df_analysis['Rain_Light'] == 1]
    df_heavy = df_analysis[(df_analysis['Rain_Moderate'] == 1) | (df_analysis['Rain_Heavy'] == 1)]

    print(f"   [!] Data Diagnostic (After Filter):")
    print(f"       - Light Rain Time-bins: {len(df_light)} records")
    print(f"       - Moderate/Heavy Rain Time-bins: {len(df_heavy)} records")

    if df_heavy.empty:
        print("\n⚠️ CẢNH BÁO: Tập dữ liệu Mưa Vừa/Lớn rỗng!")
        return None

    crisis_light = df_light.groupby('LocationID')['delta_volume'].mean().reset_index(name='avg_delta_light')
    crisis_heavy = df_heavy.groupby('LocationID')['delta_volume'].mean().reset_index(name='avg_delta_heavy')

    crisis_compare = pd.merge(crisis_light, crisis_heavy, on='LocationID', how='outer').fillna(0)

    print("5. Mapping Zone names from Shapefile...")
    gdf_zones = gpd.read_file(shapefile_path)

    # Lấy LocationID, zone và borough từ file .shp và đổi tên cho đẹp
    zone_mapping = gdf_zones[['LocationID', 'zone', 'borough']].copy()
    zone_mapping = zone_mapping.rename(columns={'zone': 'Zone', 'borough': 'Borough'})

    # Ghép bảng
    crisis_compare = pd.merge(crisis_compare, zone_mapping, on='LocationID', how='left')

    top_5_crisis = crisis_compare.nlargest(5, 'avg_delta_heavy')
    bottom_5_crisis = crisis_compare.nsmallest(5, 'avg_delta_heavy')

    print("\n" + "="*80)
    print(f"🚨 TOP 5 'CRISIS' LOCATIONS (HIGHEST SURGE) DURING MODERATE/HEAVY RAIN")
    print(f"   (Filtered: Normal demand >= {min_baseline} trips/time-bin)")
    print("="*80)
    for _, row in top_5_crisis.iterrows():
        print(f"📍 {row['Zone']} ({row['Borough']})")
        light_sign = "+" if row['avg_delta_light'] > 0 else ""
        heavy_sign = "+" if row['avg_delta_heavy'] > 0 else ""
        print(f"   - Light Rain     : {light_sign}{row['avg_delta_light']:.1f} trips/time-bin")
        print(f"   - MOD/HEAVY RAIN : {heavy_sign}{row['avg_delta_heavy']:.1f} trips/time-bin")

    print("\n" + "="*80)
    print(f"🥶 BOTTOM 5 'FROZEN' LOCATIONS (STEEPEST DROP) DURING MODERATE/HEAVY RAIN")
    print(f"   (Filtered: Normal demand >= {min_baseline} trips/time-bin)")
    print("="*80)
    for _, row in bottom_5_crisis.iterrows():
        print(f"📍 {row['Zone']} ({row['Borough']})")
        print(f"   - Light Rain     : {row['avg_delta_light']:.1f} trips/time-bin")
        print(f"   - MOD/HEAVY RAIN : {row['avg_delta_heavy']:.1f} trips/time-bin")

    return crisis_compare

if __name__ == '__main__':
    crisis_data = analyze_rain_crisis_combined(
        master_data_path='/content/drive/MyDrive/Mining of massive dataset/preprocessing_output/master_dataset_2018_clean.parquet',
        shapefile_path='/content/drive/MyDrive/Mining of massive dataset/taxi_zones/taxi_zones.shp', # SỬA LẠI ĐƯỜNG DẪN SHAPEFILE
        min_baseline=15
    )