# Code for time series analysis in Python using data from the UKHSA data dashboard. 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Data Preparation
# Data sourced from UKHSA Dashboard (Influenza Weekly Positivity)
import requests

url = "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/influenza/geography_types/Nation/geographies/England/metrics/influenza_testing_positivityByWeek"

params = {
    "format": "json",
    "page_size": 365
}

response = requests.get(url, params=params)
data = response.json()

# The results are contained in the 'results' key
df = pd.DataFrame(data['results'])

# Include manual data ingestion via csv (if required)
##df = pd.read_csv('ukhsa_influenza_positivity.csv')

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('W-MON')
df['weekly_positivity'] = df['weekly_positivity'].interpolate()

# 2. Time Series Decomposition
# Period = 52 weeks for annual seasonality
decomposition = seasonal_decompose(df['weekly_positivity'], model='additive', period=52)

# 3. Year-over-Year Analysis
df['year'] = df.index.year
df['week'] = df.index.isocalendar().week

# 4. Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['weekly_positivity'], color='#1f77b4', linewidth=2)
plt.title('UKHSA Influenza Weekly Positivity Rate (2021-2025)')
plt.ylabel('Positivity Rate (%)')
plt.grid(True, alpha=0.3)
plt.savefig('influenza_time_series.png')

# 5. Seasonal Comparison Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='week', y='weekly_positivity', hue='year', palette='viridis')
plt.title('Influenza Seasonality: Weekly Comparison by Year')
plt.xlabel('Week Number')
plt.ylabel('Positivity Rate (%)')
plt.savefig('influenza_yoy_comparison.png')
