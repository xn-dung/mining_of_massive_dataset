import numpy as np
import pandas as pd
import glob 
import os
DATA_DIR = 'D:/mining_of_massive_dataset/data/'
LOOK_UP_DIR = DATA_DIR + 'taxi_zone_lookup_grid.csv'
YEAR = 2025
BIN_FREQ = '30min'

if __name__ == '__main__':
    year_dir = os.path.join(DATA_DIR, str(YEAR))
    file_list = glob.glob(os.path.join(year_dir, '*_volume.csv'))
    df_list = [pd.read_csv(file) for file in file_list]
    df_all = pd.concat(df_list,ignore_index=True)
    col_mapping = {
        'locationid' : 'LocationID'
    }
    df_all = df_all.rename(columns=col_mapping)
    df_all['time_bin'] = pd.to_datetime(df_all['time_bin'], format='%Y-%m-%d %H:%M:%S')
    df_all = df_all[df_all['time_bin'].dt.year == YEAR].copy()

    df_lookup = pd.read_csv(LOOK_UP_DIR)
    df_merge = pd.merge(df_all,df_lookup,on="LocationID",how="left")
    missing_grid = df_merge['Grid_X'].isna() | df_merge['Grid_Y'].isna()
    if missing_grid.any():
        missing_ids = sorted(df_merge.loc[missing_grid, 'LocationID'].dropna().unique())
        raise ValueError(f"Missing grid mapping for LocationID values: {missing_ids}")

    df_merge.to_csv(os.path.join(year_dir, 'processed_data.csv'),index=False)

    C,H,W = 3,10,20

    start_time = pd.Timestamp(f'{YEAR}-01-01 00:00:00')
    end_time = pd.Timestamp(f'{YEAR + 1}-01-01 00:00:00')
    unique_times = pd.date_range(start=start_time, end=end_time, freq=BIN_FREQ, inclusive='left')

    time_to_idx = {time_val: idx for idx, time_val in enumerate(unique_times)}
    df_merge['time_idx'] = df_merge['time_bin'].map(time_to_idx)
    df_merge = df_merge.dropna(subset=['time_idx']).copy()
    df_merge['time_idx'] = df_merge['time_idx'].astype(np.int64)
    T = len(unique_times)

    df_grid = (
        df_merge
        .groupby(['time_idx', 'Grid_X', 'Grid_Y'], as_index=False)[['start_volume', 'end_volume']]
        .sum()
    )

    grid_x = df_grid['Grid_X'].astype(np.int64).values
    grid_y = df_grid['Grid_Y'].astype(np.int64).values
    time_indices = df_grid['time_idx'].values

    start_vols = df_grid['start_volume'].values
    end_vols = df_grid['end_volume'].values

    data = np.zeros((T,C,H,W), dtype = np.float32)
    data[time_indices,0,grid_x,grid_y] = start_vols
    data[time_indices,1,grid_x,grid_y] = end_vols
    data[time_indices,2, grid_x, grid_y] = start_vols - end_vols
    np.save(os.path.join(year_dir, 'taxi_volume_4d_tensor.npy'), data)


    df_holidays = pd.read_csv(os.path.join(year_dir, f'holidays_{YEAR}.csv'))
    df_weather = pd.read_csv(os.path.join(year_dir, f'weather_{YEAR}.csv'))
    df_holidays['time'] = pd.to_datetime(df_holidays['time'])
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_context_raw = pd.merge(df_holidays, df_weather, on="time", how="outer")
    df_time_sequence = pd.DataFrame({'time': unique_times})
    df_context_aligned = pd.merge(df_time_sequence, df_context_raw, on="time", how="left")
    df_context_aligned = df_context_aligned.ffill().fillna(0)
    df_features_only = df_context_aligned.drop(columns=['time'])
    context_data = df_features_only.values.astype(np.float32)
    np.save(os.path.join(year_dir, 'context_input.npy'), context_data)
    print(f"Finished! tensor={data.shape}, context={context_data.shape}, grid_rows={len(df_grid)}")

    


