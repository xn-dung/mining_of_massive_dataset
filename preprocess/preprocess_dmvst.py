import numpy as np
import pandas as pd
import glob 
import os
DATA_DIR = 'D:/mining_of_massive_dataset/data/'
LOOK_UP_DIR = DATA_DIR + 'taxi_zone_lookup_grid.csv'

if __name__ == '__main__':
    file_list = glob.glob(DATA_DIR + '*_volume.csv')
    df_list = [pd.read_csv(file) for file in file_list]
    df_all = pd.concat(df_list,ignore_index=True)
    col_mapping = {
        'locationid' : 'LocationID'
    }
    df_all = df_all.rename(columns=col_mapping)
    df_lookup = pd.read_csv(LOOK_UP_DIR)
    df_merge = pd.merge(df_all,df_lookup,on="LocationID",how="left")

    df_merge.to_csv(DATA_DIR + 'processed_data.csv',index=False)

    C,H,W = 3,10,20

    df_merge['time_bin'] = pd.to_datetime(df_merge['time_bin'],format='%Y-%m-%d %H:%M:%S')
    df_merge = df_merge.sort_values(by='time_bin')
    unique_times = sorted(df_merge['time_bin'].unique())

    time_to_idx = {time_val: idx for idx, time_val in enumerate(unique_times)}
    df_merge['time_idx'] = df_merge['time_bin'].map(time_to_idx)
    T = len(unique_times)

    grid_x = df_merge['Grid_X'].values
    grid_y = df_merge['Grid_Y'].values
    time_indices = df_merge['time_idx'].values

    start_vols = df_merge['start_volume'].values
    end_vols = df_merge['end_volume'].values

    data = np.zeros((T,C,H,W), dtype = np.float32)
    data[time_indices,0,grid_x,grid_y] = start_vols
    data[time_indices,1,grid_x,grid_y] = end_vols
    data[time_indices,2, grid_x, grid_y] = start_vols - end_vols
    np.save(DATA_DIR + 'taxi_volume_4d_tensor.npy', data)






    df_holidays = pd.read_csv(DATA_DIR + 'holidays_2018.csv')
    df_weather = pd.read_csv(DATA_DIR + "weather_2018.csv")
    df_context = pd.merge(df_holidays,df_weather,on="time",how="left")
    df_features_only = df_context.drop(columns=['time'])
    context_data = df_features_only.values.astype(np.float32)
    np.save(DATA_DIR + 'context_input.npy', context_data)
    print("Finished!")

    


