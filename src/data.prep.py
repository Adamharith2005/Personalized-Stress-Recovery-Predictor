import pandas as pd
import numpy as np

def load_and_clean_data():
    activity = pd.read_csv('data/dailyActivity_merged.csv')
    sleep = pd.read_csv('data/sleepDay_merged.csv')
    heart = pd.read_csv('data/heartrate_seconds_merged.csv')

    activity['ActivityDate'] = pd.to_datetime(activity['ActivityDate'])
    sleep['SleepDay'] = pd.to_datetime(sleep['SleepDay'], format='mixed')
    heart['Time'] = pd.to_datetime(heart['Time'], format='mixed')

    # FEATURE ENGINEERING: Calculate Resting Heart Rate (RHR)
    # Defined RHR as the average heart rate between 2 AM and 4 AM (deep sleep)
    rhr_data = heart[(heart['Time'].dt.hour >= 2) & (heart['Time'].dt.hour <= 4)]
    daily_rhr = rhr_data.groupby(rhr_data['Time'].dt.date)['Value'].mean().reset_index()
    daily_rhr.columns = ['Date', 'RestingHR']
    daily_rhr['Date'] = pd.to_datetime(daily_rhr['Date'])

    # MERGE: Combine Activity, Sleep and RHR into one table
    df = pd.merge(activity, sleep, left_on='ActivityDate', right_on='SleepDay')
    df = pd.merge(df, daily_rhr, left_on='ActivityDate', right_on='Date')

    # CALCULATE TARGET: "Recovery Score" (0-100)
    # Formula: (Sleep Ratio * 0.6) + (RHR efficiency * 0.4)
    # High sleep and low RHR is desired
    df['SleepScore'] = (df['TotalMinutesAsleep'] / 480).clip(0, 1) * 100
    df['RHRScore'] = (60 / df['RestingHR']).clip(0, 1) * 100
    df['RecoveryScore'] = (df['SleepScore'] * 0.6) + (df['RHRScore'] * 0.4)

    df = df.drop(columns=['SleepDay', 'Date'])
    
    return df

if __name__ == "__main__":
    data = load_and_clean_data()
    print(f"Success! Processed {len(data)} days of health metrics.")
    data.to_csv('data/processed_health_data.csv', index=False)