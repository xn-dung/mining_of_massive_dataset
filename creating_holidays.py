import pandas as pd

def create_holiday_features(df, date_col):
    df = df.copy()

    temp_date = pd.to_datetime(df[date_col]).dt.normalize()

    df['Is_Legal_Holiday'] = 0
    df['Is_School_Recess'] = 0
    df['Is_Event_Festival'] = 0
    df['Is_Early_Dismissal'] = 0
    

    legal_holidays = pd.to_datetime([
        '2018-01-01', # New Year's Day
        '2018-01-15', # Dr. Martin Luther King Jr. Day
        '2018-02-12', # Lincoln's Birthday
        '2018-02-19', # Washington's Birthday / Presidents' Day
        '2018-05-28', # Memorial Day
        '2018-07-04', # Independence Day
        '2018-09-03', # Labor Day
        '2018-10-08', # Columbus Day
        '2018-11-06', # Election Day
        '2018-11-12', # Veterans Day (Observed)
        '2018-11-22', # Thanksgiving Day
        '2018-12-25'  # Christmas Day
    ])
    df.loc[temp_date.isin(legal_holidays), 'Is_Legal_Holiday'] = 1
    
    # =====================================================================
    # NHÓM 2: Is_School_Recess (Học sinh nghỉ, người lớn đi làm)
    # =====================================================================
    school_recess_dates = pd.to_datetime([
        '2018-01-02', # Tiếp nối kỳ nghỉ đông 2017
        '2018-01-26', '2018-01-29', # Scoring Day / Chancellor’s Conference Day
        '2018-06-07', # Anniversary Day
        '2018-06-11', # June Clerical Day
        '2018-06-15', # Eid al-Fitr
        
        # Bổ sung thêm từ Lịch học NYC DOE (Nửa cuối năm 2018)
        '2018-09-10', '2018-09-11', # Rosh Hashanah
        '2018-09-19', # Yom Kippur
        '2018-11-23'  # Ngày sau Lễ Tạ Ơn (Black Friday)
    ]).tolist()
    
    # Thêm các chuỗi ngày nghỉ dài hạn
    school_recess_dates.extend(pd.date_range('2018-02-16', '2018-02-23').tolist()) # Lunar New Year / Midwinter
    school_recess_dates.extend(pd.date_range('2018-03-30', '2018-04-06').tolist()) # Spring Recess (Good Friday/Passover)
    school_recess_dates.extend(pd.date_range('2018-12-24', '2018-12-31').tolist()) # Winter Recess 2018-2019
    
    # LOẠI TRỪ: Nếu đã là Legal Holiday (Nhóm 1) thì không tính vào Nhóm 2 nữa
    school_recess_dates = list(set(school_recess_dates) - set(legal_holidays))
    
    df.loc[temp_date.isin(school_recess_dates), 'Is_School_Recess'] = 1
    
    # =====================================================================
    # NHÓM 3: Is_Event_Festival (Lễ hội vui chơi - Kẹt xe cục bộ)
    # =====================================================================
    events_festivals = pd.to_datetime([
        '2018-02-14', # Valentine's Day
        '2018-03-17', # St. Patrick's Day
        '2018-04-01', # Easter Sunday
        '2018-05-13', # Mother's Day
        '2018-06-17', # Father's Day
        '2018-10-31', # Halloween
        '2018-12-24'  # Christmas Eve
    ])
    # Code này không lọc theo ngày trong tuần nên nếu rơi vào T7, CN thì cờ vẫn sẽ là 1
    df.loc[temp_date.isin(events_festivals), 'Is_Event_Festival'] = 1
    
    early_dismissals = pd.to_datetime([
        '2018-03-06', # Middle School sớm
        '2018-03-09', # High School sớm
        '2018-03-13', # District 75 sớm
        '2018-03-15', # Elementary School sớm
        '2018-06-26'  # Ngày học cuối cùng (Toàn bộ về sớm)
    ])
    df.loc[temp_date.isin(early_dismissals), 'Is_Early_Dismissal'] = 1
    
    return df

# --- Cách sử dụng: ---
# Giả sử bạn đang có dataframe tên là 'df_taxi' với cột thời gian là 'tpep_pickup_datetime'
# df_taxi = create_holiday_features(df_taxi, 'tpep_pickup_datetime')