import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def temporal_analysis_monthly(volume_csv_path, lookup_csv_path, target_zone_id=161):
    """
    Function to analyze temporal trends of NYC Taxi data.
    target_zone_id=161 is Midtown Center (A bustling office district in Manhattan)
    """
    print("⏳ Loading and processing data...")
    df = pd.read_csv(volume_csv_path)
    df_lookup = pd.read_csv(lookup_csv_path)

    # Convert time_bin column to datetime format
    df['time_bin'] = pd.to_datetime(df['time_bin'])

    # Extract temporal features
    df['hour'] = df['time_bin'].dt.hour
    df['day_of_week'] = df['time_bin'].dt.dayofweek # 0: Monday, 6: Sunday
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 'Weekend (Sat, Sun)' if x >= 5 else 'Weekday (Mon-Fri)')

    # =====================================================================
    # CHART 1: DAY-OF-WEEK COMPARISON (Weekday vs Weekend)
    # =====================================================================
    plt.figure(figsize=(12, 6))

    # Calculate daily totals first, then average by hour and day type
    daily_totals = df.groupby([df['time_bin'].dt.date, 'hour', 'is_weekend'])['start_volume'].sum().reset_index()
    avg_hourly_trend = daily_totals.groupby(['hour', 'is_weekend'])['start_volume'].mean().reset_index()

    sns.lineplot(data=avg_hourly_trend, x='hour', y='start_volume', hue='is_weekend',
                 palette=['#1f77b4', '#ff7f0e'], linewidth=2.5, marker='o')

    plt.title('Traffic Rhythm Comparison: Weekday vs Weekend', fontsize=15, fontweight='bold')
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Average Trip Volume', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Day Type')
    plt.tight_layout()
    plt.show()

    # =====================================================================
    # CHART 2: TRAFFIC FLOW REVERSAL AT OFFICE DISTRICT (Time-of-day)
    # =====================================================================
    # Get the name of the target zone
    zone_name = df_lookup[df_lookup['LocationID'] == target_zone_id]['Zone'].values[0]

    plt.figure(figsize=(12, 6))

    # Filter data for the specific office zone during WEEKDAYS only
    df_office = df[(df['locationid'] == target_zone_id) & (df['is_weekend'] == 'Weekday (Mon-Fri)')]

    # Calculate the average ARRIVING (end_volume) and DEPARTING (start_volume) trips per hour
    office_flow = df_office.groupby('hour')[['start_volume', 'end_volume']].mean().reset_index()

    # Plot grouped bar chart
    bar_width = 0.35
    plt.bar(office_flow['hour'] - bar_width/2, office_flow['end_volume'],
            bar_width, label='ARRIVING (Drop-off / Commuting IN)', color='#2ca02c')
    plt.bar(office_flow['hour'] + bar_width/2, office_flow['start_volume'],
            bar_width, label='DEPARTING (Pickup / Commuting OUT)', color='#d62728')

    plt.title(f'Traffic Flow Reversal at Office District: {zone_name} (Weekdays)', fontsize=15, fontweight='bold')
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Average Number of Trips', fontsize=12)
    plt.xticks(range(0, 24))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # =====================================================================
    # CHART 3: NET FLOW (THE CORE INSIGHT)
    # =====================================================================
    plt.figure(figsize=(12, 5))

    # Net Flow = Arriving - Departing
    # > 0 : The area is "Absorbing" people (Sink)
    # < 0 : The area is "Pushing" people out (Source)
    office_flow['net_flow'] = office_flow['end_volume'] - office_flow['start_volume']

    colors = ['#2ca02c' if x > 0 else '#d62728' for x in office_flow['net_flow']]
    plt.bar(office_flow['hour'], office_flow['net_flow'], color=colors)
    plt.axhline(0, color='black', linewidth=1.5)

    plt.title(f'Net Traffic Flow at {zone_name}', fontsize=15, fontweight='bold')
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Net Flow (Arriving - Departing)', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.text(8, max(office_flow['net_flow'])*0.8, 'Morning: Massive Influx of Workers', fontsize=11, color='green', fontweight='bold')
    plt.text(17, min(office_flow['net_flow'])*0.8, 'Evening: Massive Exodus', fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()

temporal_analysis_monthly('/content/drive/MyDrive/Mining of massive dataset/preprocessing_output/yellow_data/yellow_2023/01_volume.csv', '/content/drive/MyDrive/Mining of massive dataset/taxi_zone_lookup.csv', target_zone_id=161)
# plot_volume_with_insights('/content/drive/MyDrive/Mining of massive dataset/taxi_zones/taxi_zones.shp', '/content/drive/MyDrive/Mining of massive dataset/preprocessing_output/yellow_data/yellow_2023/01_volume.csv', '/content/drive/MyDrive/Mining of massive dataset/taxi_zone_lookup.csv')