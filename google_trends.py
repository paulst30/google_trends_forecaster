# import packages
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import time
import datetime
from darts import TimeSeries
from matplotlib import pyplot as plt
from darts.metrics import mape
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Prophet
import argparse

# functions

# get data from google
def get_data(keyword, cat):
  run = 0
  wait = [2, 4, 6, 8]
  pt = TrendReq(hl='de-DE', tz=60)
  de_data = None

  while de_data is None:
    try:
      pt.build_payload(keyword, cat=cat, timeframe='today 5-y', geo='DE')
      de_data = pt.interest_over_time()
      de_data = de_data.drop(columns='isPartial')

    except Exception as e:
      if run == 4:
         break
      print('Google is rejecting us...')
      print('Retrying in', wait[run], 'seconds...')
      time.sleep(wait[run])
      run += 1

  return de_data


# data prep
def data_prep(data, keyword, split_date):
  data = data.reset_index()
  ts_data = TimeSeries.from_dataframe(data, 'date', keyword)
  train,val = ts_data.split_after(pd.Timestamp(split_date))
  return ts_data , train, val


# find best model
def model_comparison(data, models, keyword, split_date):

  # setup
  performances = dict()
  best_mape = np.Inf

  # compare models
  for modelx in models:
    model = modelx

    hfc_params = {
    "series": data,
    "start": pd.Timestamp(split_date),
    "forecast_horizon": 1,
    "verbose": True,
    }

    # expanding window backtest
    hist_model = model.historical_forecasts(last_points_only=True, **hfc_params)

    #calculate and track metrics
    mapex = mape(data, hist_model)
    model_name = str(model).split("(")[0]
    performances.update({model_name : mapex})

  # update best model and extract residuals
  if mapex < best_mape:
    best_mape = mapex
    best_model = model

    back_forecast = hist_model.pd_dataframe()
    residuals = (data.pd_dataframe() - back_forecast)
    residuals = residuals.dropna(subset=[keyword[0]])
    print("found best model")
  return performances, best_model, residuals


# train model on all the data
def model_production(data, model):

  trained_model = model
  trained_model.fit(data)
  print("successfully trained model")
  #potentially save the model here

  return trained_model


# predict next 60 weeks
def model_prediction(trained_model):

  forecast = trained_model.predict(n=60)
  forecast = forecast.pd_dataframe()

  return forecast

# generate output files (csv and graphs)
def gen_output(data, forecast, residuals):
  output = pd.concat([data, forecast, residuals], axis=1)
  output.columns =['series', 'forecast', 'residuals']
  output.index.name = 'date'
  output['train'] = output['series'] + output['residuals']

  # save output as csv
  output.to_csv('output.csv', index=True)

  # generate graphs
  fig, axs = plt.subplots(3, 2, figsize=(15, 20))

  # forecast
  ax = fig.add_subplot(3, 1, 1)  # Create a new subplot spanning the first row
  ax.plot(output['series'], label='Series')
  ax.plot(output['forecast'], label='Forecast')
  ax.set_title('Forecast')
  ax.legend()

  # Remove the empty axes (axs[0, 0] and axs[0, 1])
  fig.delaxes(axs[0, 0])
  fig.delaxes(axs[0, 1])
  
  # training data and training prediction
  ax = fig.add_subplot(3, 1, 2)  # Create a new subplot spanning the first row
  ax.plot(output['series'], label='Train')
  ax.plot(output['train'], label='Prediction')
  ax.set_title('Training data and residuals')
  ax.legend()

  # Remove the empty axes (axs[0, 0] and axs[0, 1])
  fig.delaxes(axs[1, 0])
  fig.delaxes(axs[1, 1])
  
  #plot residuals
  axs[2,0].plot(output['residuals'])
  axs[2,0].set_title('Residuals over time')

  axs[2,1].hist(output['residuals'])
  axs[2,1].set_title('Distribution of residuals')

  #save graph
  fig.savefig('output.png')


#### Main function

# main function
def main(keyword, cat, models, split_date):
    # Step 1: Download data
    print('Downloading data...')
    data = get_data(keyword, cat)

    # Step 2: Data preparation
    print('Preparing data...')
    ts_data , train, val = data_prep(data, keyword, split_date)

    # Step 2: Model comparison
    performances, best_model, residuals = model_comparison(data = ts_data, models=models, keyword = keyword, split_date = split_date)

    # Step 3: Model production (train the best model)
    trained_model = model_production(data=ts_data, model=best_model)

    # Step 4: Make predictions
    forecast = model_prediction(trained_model=trained_model)

    # Step 5: generate output files
    gen_output(data, forecast, residuals)

## direct script call
if __name__ == "__main__":

  # argument parser
  parser = argparse.ArgumentParser()

  # input arguments
  parser.add_argument('--keyword', type=str, required=True, help='Keyword for google search')
  parser.add_argument('--cat', type=str, required=True, help='Comparison catgory for keyword')
  parser.add_argument('--models', default = [ExponentialSmoothing()], help='List of models to compare')

  args = parser.parse_args()

  keyword = args.keyword
  cat = args.cat
  models = args.models

  # constants
  date = datetime.datetime.now() - datetime.timedelta(days=365)
  split_date = date.strftime('%Y-%m-%d')
  
  # run script
  main(keyword = keyword, cat = cat, models = models, split_date = split_date)
