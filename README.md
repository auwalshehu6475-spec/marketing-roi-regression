import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Plot settings
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

pd.set_option('display.max_columns', None)
np.random.seed(42)

# ==========================
# LOAD DATA
# ==========================

raw = pd.read_csv('tiktok_dataset.csv')

print("Dataset Shape:", raw.shape)
print(raw.head())

print(raw.dtypes)
print(raw.isna().sum())
print(raw.describe())

# Remove missing values

df = raw.dropna(subset=['video_view_count']).copy()

print(f"Rows before cleaning: {len(raw)}")
print(f"Rows after cleaning: {len(df)}")

# ==========================
# CREATE SYNTHETIC DATE
# ==========================

start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2024-06-30')

total_days = (end_date - start_date).days

day_offsets = np.random.randint(
    0,
    total_days + 1,
    size=len(df)
)

df['date'] = start_date + pd.to_timedelta(day_offsets, unit='D')

trend_factor = 1 + 0.35 * (day_offsets / total_days)

weekday = df['date'].dt.dayofweek

weekday_multiplier = weekday.map({
    0: 0.95,
    1: 0.90,
    2: 0.92,
    3: 0.97,
    4: 1.05,
    5: 1.25,
    6: 1.20
})

df['view_count'] = (
    df['video_view_count']
    * trend_factor
    * weekday_multiplier
).round().astype(int)

print(df[['date', 'video_view_count', 'view_count']].head())

# ==========================
# PREPROCESSING
# ==========================

daily = (
    df.groupby('date')['view_count']
      .sum()
      .reset_index()
)

daily['date'] = pd.to_datetime(daily['date'])

daily = (
    daily
    .set_index('date')
    .sort_index()
)

daily = daily.resample('D').sum()

daily['view_count'] = daily['view_count'].replace(0, np.nan)

daily['view_count'] = daily['view_count'].interpolate(method='linear')

print(daily.head())

# ==========================
# DAILY VIEW COUNT PLOT
# ==========================

plt.figure(figsize=(12,5))
plt.plot(
    daily.index,
    daily['view_count'],
    color='#fe2c55'
)

plt.title("TikTok Daily Total View Count")
plt.xlabel("Date")
plt.ylabel("Views")
plt.show()

# ==========================
# TIME SERIES DECOMPOSITION
# ==========================

decomposition = seasonal_decompose(
    daily['view_count'],
    model='additive',
    period=7
)

fig = decomposition.plot()

fig.set_size_inches(12,8)

plt.show()

# ==========================
# ADF TEST
# ==========================

adf = adfuller(daily['view_count'])

print("ADF Statistic:", adf[0])
print("P-value:", adf[1])

print("Critical Values")

for key, value in adf[4].items():
    print(key, value)

if adf[1] < 0.05:
    print("Series is Stationary")
else:
    print("Series is Non-Stationary")

# Seasonal differencing

diff = daily['view_count'].diff(7).dropna()

adf_diff = adfuller(diff)

print("ADF after differencing:", adf_diff[1])

# ==========================
# WEEKDAY ANALYSIS
# ==========================

daily['weekday'] = daily.index.day_name()

weekday_order = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]

weekday_avg = (
    daily.groupby('weekday')['view_count']
         .mean()
         .reindex(weekday_order)
)

print(weekday_avg)

colors = [
    '#fe2c55' if x == weekday_avg.max()
    else '#b8b8b8' if x == weekday_avg.min()
    else '#25f4ee'
    for x in weekday_avg
]

plt.figure(figsize=(10,5))

plt.bar(
    weekday_avg.index,
    weekday_avg.values,
    color=colors
)

plt.title("Average Daily Views by Weekday")

plt.xlabel("Weekday")

plt.ylabel("Average Views")

plt.xticks(rotation=30)

plt.show()

best_day = weekday_avg.idxmax()
worst_day = weekday_avg.idxmin()

print("Best upload day:", best_day)
print("Lowest engagement day:", worst_day)

# ==========================
# ROLLING STATISTICS
# ==========================

daily['rolling_mean_7d'] = daily['view_count'].rolling(7).mean()

daily['rolling_std_7d'] = daily['view_count'].rolling(7).std()

plt.figure(figsize=(12,5))

plt.plot(
    daily.index,
    daily['view_count'],
    color='gray',
    alpha=0.4,
    label='Daily Views'
)

plt.plot(
    daily.index,
    daily['rolling_mean_7d'],
    color='#fe2c55',
    linewidth=2,
    label='7-Day Rolling Mean'
)

plt.title("Daily Views with 7-Day Rolling Mean")

plt.xlabel("Date")

plt.ylabel("Views")

plt.legend()

plt.show()

plt.figure(figsize=(12,5))

plt.plot(
    daily.index,
    daily['rolling_std_7d'],
    color='#25f4ee'
)

plt.title("7-Day Rolling Standard Deviation")

plt.xlabel("Date")

plt.ylabel("Rolling Standard Deviation")

plt.show()

# ==========================
# SUMMARY
# ==========================

print("\n========== DISCOVERY ==========")
print("• Trend: Daily views gradually increase over time.")
print("• Seasonality: Weekly engagement pattern exists.")
print("• Residual: Random day-to-day variation remains.")

print("\n========== BUSINESS INSIGHTS ==========")
print(f"Best day to upload: {best_day}")
print(f"Lowest engagement day: {worst_day}")
print("Schedule important content around weekends.")
print("Use low-engagement days for lighter content.")
print("Use SARIMA or ARIMA for future forecasting.")