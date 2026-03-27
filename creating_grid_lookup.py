import geopandas as gpd
import pandas as pd
import numpy as np

def create_grid_lookup(shapefile_path, output_csv):
    print("Đang đọc dữ liệu Shapefile...")
    zones = gpd.read_file(shapefile_path)

    zones = zones.to_crs(epsg=4326)

    zones['centroid_lon'] = zones.geometry.centroid.x
    zones['centroid_lat'] = zones.geometry.centroid.y

    lon_min, lon_max = zones['centroid_lon'].min(), zones['centroid_lon'].max()
    lat_min, lat_max = zones['centroid_lat'].min(), zones['centroid_lat'].max()
    print(f"Lon: [{lon_min:.4f}, {lon_max:.4f}], Lat: [{lat_min:.4f}, {lat_max:.4f}]")

    GRID_X_BINS = 10
    GRID_Y_BINS = 20

    grid_x_size = (lon_max - lon_min) / GRID_X_BINS
    grid_y_size = (lat_max - lat_min) / GRID_Y_BINS

    zones['Grid_X'] = np.floor((zones['centroid_lon'] - lon_min) / grid_x_size).astype(int)
    zones['Grid_Y'] = np.floor((zones['centroid_lat'] - lat_min) / grid_y_size).astype(int)

    zones['Grid_X'] = zones['Grid_X'].clip(0, GRID_X_BINS - 1)
    zones['Grid_Y'] = zones['Grid_Y'].clip(0, GRID_Y_BINS - 1)

    lookup_table = zones[['LocationID', 'Grid_X', 'Grid_Y']]

    lookup_table.to_csv(output_csv, index=False)
    print(f"Hoàn tất, lưu tại: {output_csv}")