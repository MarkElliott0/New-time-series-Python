# Code for time series analysis in Python using data from the UKHSA data dashboard. 
# Created with support from Google Gemini 
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def fetch_influenza_data():
    """Fetches live influenza positivity data from the UKHSA API."""
    # API Endpoint for England-wide Influenza Testing Positivity
    api_url = "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/influenza/geography_types/Nation/geographies/England/metrics/influenza_testing_positivityByWeek"
    
    params = {
        "format": "json",
        "page_size": 1000  # Captures the full available time series
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        json_data = response.json()
        
        # Convert JSON results to DataFrame
        df = pd.DataFrame(json_data['results'])
        
        # Format the date and sort chronologically
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'metric_value': 'positivity_rate'})
        df = df.sort_values('date')
        
        return df[['date', 'positivity_rate']]
    
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None

# --- Main Analysis Execution ---
df = fetch_influenza_data()

if df is not None:
    # Set index to datetime for time series analysis
    df.set_index('date', inplace=True)
    
    # Ensure regular weekly frequency (UKHSA weeks usually report on Mondays)
    df = df.asfreq('W-MON')
    df['positivity_rate'] = df['positivity_rate'].interpolate()

    # 1. Plotting the Live Trend
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['positivity_rate'], color='#2a9d8f', linewidth=2, label='Weekly Positivity')
    plt.fill_between(df.index, df['positivity_rate'], color='#2a9d8f', alpha=0.1)
    
    plt.title('Live Time Series: UK Influenza Weekly Positivity Rate', fontsize=14)
    plt.ylabel('Positivity Rate (%)')
    plt.xlabel('Year')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

    # 2. Seasonal Decomposition (52-week period)
    result = seasonal_decompose(df['positivity_rate'], model='additive', period=52)
    result.plot()
    plt.suptitle('Seasonal Decomposition of UK Influenza Data', y=1.02)
    plt.show()

    # 3. Monthly Heatmap (Annual Trends)
    df['year'] = df.index.year
    df['month'] = df.index.month_name()
    pivot_table = df.pivot_table(values='positivity_rate', index='month', columns='year', aggfunc='mean')
    
    # Sort months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    pivot_table = pivot_table.reindex(month_order)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt=".1f")
    plt.title('Mean Monthly Influenza Positivity (%)')
    plt.show()

else:
    print("Data fetch failed. Please check your internet connection or the API URL.")
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def fetch_influenza_data():
    """Fetches live influenza data and handles the nested JSON structure."""
    api_url = "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/influenza/geography_types/Nation/geographies/England/metrics/influenza_testing_positivityByWeek"
    
    params = {"format": "json", "page_size": 1000}
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Ensure 'results' exists to avoid KeyError
        if 'results' not in data:
            print("Error: 'results' key not found in API response.")
            return None
            
        df = pd.DataFrame(data['results'])
        
        # Prophet requires columns named 'ds' (date) and 'y' (value)
        # We use .get() to safely check for column names
        date_col = 'date' if 'date' in df.columns else 'dt' 
        value_col = 'metric_value' if 'metric_value' in df.columns else 'value'
        
        df = df[[date_col, value_col]].copy()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        
        return df.sort_values('ds')
    
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None

# --- Execution ---
df = fetch_influenza_data()

if df is not None:
    # 1. Initialize and Fit the Prophet Model
    # We enable yearly seasonality as flu is highly seasonal
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)

    # 2. Create Future Dates (Forecast for the next 12 weeks)
    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)

    # 3. Visualization
    fig1 = model.plot(forecast)
    plt.title('Influenza Positivity Forecast: UKHSA Data', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Positivity Rate (%)')
    plt.axvline(df['ds'].max(), color='red', linestyle='--', label='Forecast Start')
    plt.legend()
    plt.show()

    # 4. Decomposition of Components (Trend vs Seasonality)
    fig2 = model.plot_components(forecast)
    plt.show()

    print(f"Forecast for the next 4 weeks:\n{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4)}")