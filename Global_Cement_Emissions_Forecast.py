import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset into a DataFrame
# Change the path according to your local machine 
df = pd.read_csv('Datasets/annual-co2-cement.csv')

# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Summary of the DataFrame
print("\nSummary of the DataFrame:")
print(df.info())

# Summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Rename columns
df = df.rename(columns={'Entity': 'Entity',
                        'Code': 'Entity_Code',
                        'Year': 'Year',
                        'Annual CO₂ emissions from cement': 'Annual_Emissions'})
# Display the first few rows of the DataFrame to verify the changes
print(df.head())

# Filter data for Entity_Code = 'OWID_WRL' and years between 1950 and 2022
filtered_df_world = df[(df['Entity_Code'] == 'OWID_WRL') & (df['Year'] >= 1950) & (df['Year'] <= 2022)]

# Filter data for each continent
continents = ['Africa', 'Asia', 'Australia', 'Europe', 'North America', 'South America']

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot for continents
axes[0].set_title('Trends in Annual Emissions (Continents)')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Annual Emissions')
axes[0].grid(True)
for continent in continents:
    filtered_df_continent = df[(df['Entity'] == continent) & (df['Year'] >= 1950) & (df['Year'] <= 2022)]
    axes[0].plot(filtered_df_continent['Year'], filtered_df_continent['Annual_Emissions'], marker='o', markersize=3, linestyle='-', label=continent)
axes[0].legend()

# Plot for world
axes[1].plot(filtered_df_world['Year'], filtered_df_world['Annual_Emissions'], marker='o', markersize=3, linestyle='-', label='World')
axes[1].set_title('Trends in Annual Emissions (World)')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Annual Emissions')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

# Plot KDE of annual emissions
plt.figure(figsize=(8, 6))
sns.kdeplot(filtered_df_world['Annual_Emissions'], color='blue', shade=True)
plt.title('Kernel Density Estimation (KDE) of Annual Emissions (World)')
plt.xlabel('Annual Emissions')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

# Filter data for specific income groups
income_groups = ['High-income countries', 'Low-income countries', 'Lower-middle-income countries']
filtered_df = df[df['Entity'].isin(income_groups)]

# Filter data for the year range 2010-2022
filtered_df = filtered_df[(filtered_df['Year'] >= 2010) & (filtered_df['Year'] <= 2022)]

# Pivot the DataFrame to create a grouped bar chart
pivot_df = filtered_df.pivot_table(index='Year', columns='Entity', values='Annual_Emissions', aggfunc='sum')

# Plot the grouped bar chart
pivot_df.plot(kind='bar', stacked=False, figsize=(12, 6))
plt.title('Annual CO2 Emissions by Income Group (2010-2022)')
plt.xlabel('Year')
plt.ylabel('Annual CO2 Emissions')
plt.xticks(rotation=45)
plt.legend(title='Income Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

########################################################################################################################################################################################

###### ARIMA FORECAST

# Convert 'Year' column to datetime format
filtered_df_world['Year'] = pd.to_datetime(filtered_df_world['Year'], format='%Y')

# Set 'Year' column as index
filtered_df_world.set_index('Year', inplace=True)

print('Check point 1')

# Fit ARIMA model
model = ARIMA(filtered_df_world['Annual_Emissions'], order=(5,1,0))
model_fit = model.fit()

print('Check point 2')

# Forecast CO2 emissions for the next 28 years (2023 to 2050)
forecast = model_fit.forecast(steps=28)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(filtered_df_world.index, filtered_df_world['Annual_Emissions'], label='Historical Data', color='blue')
plt.plot(pd.date_range(start='2023', periods=28, freq='Y'), forecast, label='Forecast', color='green')
plt.title('Forecast of Annual CO₂ emissions from cement')
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions (million tonnes)')
plt.legend()
plt.grid(True)
plt.show()

########################################################################################################################################################################################

########################################################################################################################################################################################

###### ETS

# Data Preprocessing
filtered_df_world['Year'] = pd.to_datetime(filtered_df_world['Year'], format='%Y')
filtered_df_world.set_index('Year', inplace=True)

# Model Fitting
model = ExponentialSmoothing(filtered_df_world['Annual_Emissions'], trend='add', seasonal='add', seasonal_periods=12)
fitted_model = model.fit()

# Forecasting (Change 5 to the number of years you want to forecast)
forecast = fitted_model.forecast(steps=27)

# Plot the historical data and forecasted emissions
plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(filtered_df_world.index, filtered_df_world['Annual_Emissions'], label='Historical Data', color='blue')

# Plot forecasted emissions
forecast_index = pd.date_range(start='2023', periods=27, freq='Y')
plt.plot(forecast_index, forecast, label='Forecast', color='green')

# Highlight the forecast period
plt.axvspan(forecast_index[0], forecast_index[-1], color='gray', alpha=0.2)

# Adding labels and legend
plt.title('Historical and Forecasted Annual CO₂ emissions from cement')
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions (million tonnes)')
plt.legend()
plt.grid(True)
plt.show()

########################################################################################################################################################################################