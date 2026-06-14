import pandas as pd
import os
import holidays
from dateutil.easter import easter
from dateutil.relativedelta import relativedelta, SU

def process_dynamic_holidays(years, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        print(f"\nĐang khởi tạo ma trận Holidays cho năm {year}...")

        time_index = pd.date_range(start=f'{year}-01-01 00:00:00',
                                   end=f'{year}-12-31 23:30:00',
                                   freq='30min')
        holidays_df = pd.DataFrame(index=time_index)

        ny_holidays = holidays.US(state='NY', years=year)
        legal_holidays = [date.strftime('%Y-%m-%d') for date in ny_holidays.keys()]

        fixed_events = [f'{year}-02-14', f'{year}-10-31', f'{year}-12-24', f'{year}-12-31']
        easter_date = easter(year)
        mothers_day = pd.to_datetime(f'{year}-05-01') + relativedelta(weekday=SU(+2))
        fathers_day = pd.to_datetime(f'{year}-06-01') + relativedelta(weekday=SU(+3))

        event_festivals = fixed_events + [
            easter_date.strftime('%Y-%m-%d'),
            mothers_day.strftime('%Y-%m-%d'),
            fathers_day.strftime('%Y-%m-%d')
        ]

        def get_date_range(start, end, y=year):
            return pd.date_range(start=f'{y}-{start}', end=f'{y}-{end}').strftime('%Y-%m-%d').tolist()

        easter_str = easter_date.strftime('%Y-%m-%d')
        spring_break_start = (easter_date - pd.Timedelta(days=5)).strftime('%m-%d')
        spring_break_end = (easter_date + pd.Timedelta(days=2)).strftime('%m-%d')

        school_recess = (
            get_date_range('02-18', '02-23') +
            get_date_range(spring_break_start, spring_break_end) +
            get_date_range('12-24', '12-31')
        )

        date_strs = holidays_df.index.strftime('%Y-%m-%d')

        holidays_df['Is_Legal_Holiday'] = date_strs.isin(legal_holidays).astype(int)
        holidays_df['Is_School_Recess'] = date_strs.isin(school_recess).astype(int)
        holidays_df['Is_Event_Festival'] = date_strs.isin(event_festivals).astype(int)

        output_path = os.path.join(output_dir, f'holidays_{year}_processed.csv')
        holidays_df.to_csv(output_path)
        print(f"-> Đã lưu lịch thông minh tại: {output_path}")

if __name__ == "__main__":
    TARGET_YEARS = range(2018, 2025)
    OUTPUT_DIR = "/content/drive/MyDrive/Mining of massive dataset/final_code/DVC_storage/data/context"

    process_dynamic_holidays(TARGET_YEARS, OUTPUT_DIR)