# Weather-Prediction-CNN_BiLSTM

A deep learning model combining CNN (Convolutional Neural Networks) and BiLSTM (Bidirectional Long Short-Term Memory) for temperature prediction based on historical weather data.

## Overview

This project uses a hybrid architecture to predict temperature based on 24-hour historical weather data sequences. The model combines the spatial feature extraction capabilities of CNNs with the sequential learning abilities of BiLSTMs to create a powerful forecasting model.

## Features

- **Hybrid CNN-BiLSTM Architecture**: Combines CNN for pattern extraction and BiLSTM for temporal relationships
- **Packaged as a Python Library**: Can be easily reused on any weather dataset
- **Time Series Processing**: Creates sequential data from time series data
- **Data Preprocessing**: Handles data cleaning, normalization, and feature extraction

## Project Structure

- `weathercnn_bilstm/` - Core Python package
- `examples/` - Example usage
- `Weather-Prediction-CNN_BiLSTM.ipynb` - Original notebook with implementation
- `Weather-Prediction-CNN_BiLSTM-v2.ipynb` - Improved implementation with fixed prediction function

## Installation

```bash
# Install the package locally
pip install -e .
```

## Usage

```python
from weathercnn_bilstm import WeatherPredictor, WeatherDataProcessor

# Initialize data processor
processor = WeatherDataProcessor(sequence_length=24)

# Load and preprocess weather data
df = processor.load_data("weather_data.csv")
df = processor.preprocess_data(df, date_col='timestamp_column')

# Create sequences for model
X, y, feature_cols = processor.create_sequences(df)

# Split data for training
X_train, X_val, X_test, y_train, y_val, y_test = processor.train_val_test_split(X, y)

# Create and train model
model = WeatherPredictor()
model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
model.train(X_train, y_train, X_val, y_val)

# Evaluate model
results = model.evaluate(X_test, y_test, processor, feature_cols)
print(f"RMSE: {results['rmse']:.4f}")
```

## Data Requirements

The model works with weather data that includes:
- A datetime column that can be parsed
- A temperature column named 'Temperature (C)' (can be adjusted in code)
- Additional numeric features representing weather variables

## License

MIT License