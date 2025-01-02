# Google Trend Forecasting
This project provides a Python script to download and analyze Google Trends data for a specific keyword, compare multiple time-series forecasting models, and predict future search interest. It generates visualizations and CSV outputs summarizing the results.

## Features

- Automatically retrieves Google Trends data for the last 5 years.

- Prepares the data for time-series analysis.

- Compares models (e.g., Exponential Smoothing, Prophet, AutoARIMA).

- Predicts the next 60 weeks of search interest.

- Generates CSV outputs and visualizations, including forecast, actual data, and residual plots.

## Installation

### Clone the repository:

```bash
git clone https://github.com/paulst30/google_trends_forecaster.git
cd google_trends_forecaster
```

### Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate    # For Windows
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the Script. Use the following command to execute the script from the command line:
```bash
python google_trends.py --keyword "['<YourKeyword>']" --cat '<CategoryCode>' --models "[<YourModels>]"
```

`--keyword`: The keyword to analyze (e.g., `['Praktikum']`).

`--cat`: The category code for comparison (e.g., `'958'`). A list of available categories can be found [here](https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories).

`--models`: A list of models to compare (e.g., `[ExponentialSmoothing()]`). The following models from the darts library are available: ExponentialSmoothing, TBATS, AutoARIMA, Prophet.

Example:

```bash
python google_trends.py --keyword "['Praktikum']" --cat '958' --models "[ExponentialSmoothing(), TBATS()]"
```

### Outputs

The script generates:

- `output.csv`: Contains the original data, forecast, residuals, and training data.

- `output.png`: Visualization of the forecast, residuals, and training data.

### Dependencies

Python v3.10.12

The script uses the following libraries:

darts == 0.32.0

pytrends == 4.9.2

pandas == 2.2.2

numpy == 1.26.4

matplotlib == 3.8.0
