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
    api_url = "hhttps://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory"
    
    params = {
        "format": "json",
        "page_size": 1000  # Captures the full available time series
    }
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        json_data = response.json()
        
        json_data

        # Convert JSON results to DataFrame
        df = pd.DataFrame(json_data['results'])
        df


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




## COVID 19 analysis 
import requests
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def fetch_ukhsa_active_data():
    # VERIFIED 2025 ENDPOINT: COVID-19 cases count (Rolling Mean) for England
    # This metric is widely populated and excellent for trend analysis.
    url = "https://api.ukhsa-dashboard.data.gov.uk/themes/infectious_disease/sub_themes/respiratory/topics/COVID-19/geography_types/Nation/geographies/England/metrics/COVID-19_cases_countRollingMean"
    
    # Requesting the max page size (365 days) to ensure a solid dataframe
    try:
        response = requests.get(url, params={"page_size": 365}, timeout=10) # check for a response from the api 
        response.raise_for_status() # raise for the status of the api
        data = response.json() # define variable data from the response of the api call
        
        results = data.get('results', []) # pull the results 
        if not results:
            print("Alert: API reached but no data found for this specific metric.") # if unable to pull data, print that the API has been reached but no data found.
            return None # return nothing if no data found
            
        df = pd.DataFrame(results) # Create a dataframe from the results shown above (if data is found)
        
        # Data Cleaning for the time series analysis
        df['date'] = pd.to_datetime(df['date']) # Change the date column to a datetime variable 
        df['metric_value'] = pd.to_numeric(df['metric_value']) # Change the metric value column to a numeric 
        df = df.sort_values('date').set_index('date') # Sort the values in the dataframe by date
        
        # Resample to ensure daily frequency (fills missing dates if any)
        df = df['metric_value'].resample('D').mean().interpolate() # create an interpolation for missing fields based on the mean of previous samples
        
        return df # return the dataframe 
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None # else return nothing and print that there was an error fetching data. 

# --- RUN ANALYSIS ---
ts = fetch_ukhsa_active_data() # Create the ts dataset for the analysis

if ts is not None:
    print(f"Success! Loaded {len(ts)} days of data.") # if the ts variable is not none, print that there has been success loading (number of rows) days worth of data
    
    # 1. Plotting the Raw Time Series
    plt.figure(figsize=(12, 5)) # Create the plot size for the figure
    ts.plot(title="COVID-19 Case Rolling Mean (England)", color="#2c3e50", lw=2) # Create the plot title 
    plt.ylabel("Value") # change the label of the y-axis to 'Value'
    plt.grid(True, alpha=0.3) # Plot the grid with an alpha value of 0.3 
    plt.show() # Print the plot

    # 2. Seasonal Decomposition (Looking for weekly patterns)
    # Using period=7 because health data usually follows a 7-day reporting cycle
    analysis = seasonal_decompose(ts, model='additive', period=7) # create an additive model, using a period of 7-days
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True) # Create smaller sub-plots (on the same visual) based on the outputs of axis 1, 2 and 3.
    analysis.trend.plot(ax=ax1, title="1. Trend (Long-term direction)", color="blue") # add a title to plot 1 and colour in blue
    analysis.seasonal.plot(ax=ax2, title="2. Seasonality (Weekly reporting effects)", color="green") # add title to plot 2 and colour in green
    analysis.resid.plot(ax=ax3, title="3. Residuals (Unexplained noise/shocks)", color="red", ls='', marker='.') # add title to plot 3 and colour in red
    plt.tight_layout() # use the tight layout funtion to fit within a single visual
    plt.show() # Show the output
else:
    print("DataFrame is still empty. Please check your internet connection or API status.") # Else, if the function fails to find data, print dataframe is still empty. 
    