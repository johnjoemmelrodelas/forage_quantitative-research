# Import Libraries

# Data Transformations
import numpy as np
import pandas as pd
import itertools
import warnings

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Model Building
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.statespace.sarimax import SARIMAX


class NatGas_TSA():
    def __init__(self, filename):
        self.filename = filename
        self.y, self.y_train, self.y_test = self.Clean_Data()
        self.model = self.Create_Model()

    def Clean_Data(self):
        # load File
        base_df = pd.read_csv('nat_gas_data.csv')
        clean_df = base_df.copy() # Preserve original loaded dataset through copy
        clean_df['Dates'] = pd.to_datetime(clean_df['Dates'], format='%m/%d/%y') # Parse date column from object to datetime
        clean_df = clean_df.set_index('Dates') 
        y = clean_df['Prices'] # Load Prices column as a series with index of dates
        
        # Split data into train and test
        cutoff = 0.75
        y_train = y.iloc[:int(len(y) * cutoff)]
        y_test = y.iloc[int(len(y) * cutoff):]
        
        return y, y_train, y_test
    
    def Create_Model(self):      

        y_pred_wfv = pd.Series()
        history = self.y_train.copy()
        for i in range(len(self.y_test)):
            warnings.filterwarnings("ignore")
            model = SARIMAX(history, order=(1, 1, 0), seasonal_order=(2, 0, 0, 12)).fit()
            next_pred = model.forecast()
            y_pred_wfv = pd.concat([y_pred_wfv,next_pred])
            history = pd.concat([history,self.y_test[next_pred.index]])
            
        return model
    
    def Predict_Price(self, date):
        # Define the target date
        target_date = pd.to_datetime(date)

        # Ensure date is beyond the training set
        last_date = self.y_train.index[-1]
        n_steps = (target_date.to_period('M') - last_date.to_period('M')).n + 1

        # Forecast steps ahead
        forecast = self.model.get_forecast(steps=n_steps)

        # Extract forecasted values
        forecast_values = forecast.predicted_mean
        forecast_conf_int = forecast.conf_int()

        # Get prediction for the target date
        predicted_value = forecast_values[target_date]
        conf_interval = forecast_conf_int.loc[target_date]

        print(f"Predicted value for {target_date.date()}: {predicted_value:.2f}")
        print(f"95% confidence interval: {conf_interval.values}")

        forecast_index = forecast_values.index
        plt.figure(figsize=(15, 5))
        plt.plot(self.y, label='Historical')
        plt.plot(forecast_values, label='Forecast', color='orange')
        plt.fill_between(
            forecast_index,
            forecast_conf_int.iloc[:, 0],
            forecast_conf_int.iloc[:, 1],
            color='orange', alpha=0.3
        )    
            
        plt.axvline(target_date, color='red', linestyle='--', label='Target Date')
        plt.legend()
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.title("SARIMAX Forecast")
        
        return plt.show()