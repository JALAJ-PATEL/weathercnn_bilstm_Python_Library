# WeatherCNN_BiLSTM

A Python library for temperature prediction using CNN-BiLSTM hybrid neural networks.

## Features

- **Hybrid CNN-BiLSTM Architecture**: Combines Convolutional Neural Networks with Bidirectional LSTM
- **Complete Data Pipeline**: Includes data loading, preprocessing, scaling, and sequence generation
- **Easy-to-use API**: Simple interface for training models and making predictions
- **Pretrained Model Support**: Load and use previously trained models
- **Customizable**: Adjust sequence length, model parameters, and more

## Installation

```bash
pip install weathercnn_bilstm
```

Or install directly from the source:

```bash
git clone https://github.com/yourusername/weathercnn_bilstm.git
cd weathercnn_bilstm
pip install -e .
```

## Quick Start

```python
from weathercnn_bilstm import WeatherPredictor, WeatherDataProcessor

# Initialize data processor
processor = WeatherDataProcessor(sequence_length=24)

# Load and preprocess data
df = processor.load_data("weather_data.csv")
df = processor.preprocess_data(df, date_col='timestamp_column')

# Create sequences
X, y, feature_cols = processor.create_sequences(df)

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = processor.train_val_test_split(X, y)

# Build and train the model
model = WeatherPredictor()
model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
model.train(X_train, y_train, X_val, y_val)

# Evaluate the model
results = model.evaluate(X_test, y_test, processor, feature_cols)
print(f"RMSE: {results['rmse']:.4f}")

# Save the model
model.save_model("my_weather_model.keras")
```

## Using a Pre-trained Model

```python
from weathercnn_bilstm import WeatherPredictor, WeatherDataProcessor

# Load the model
model = WeatherPredictor.load_from_file("my_weather_model.keras")

# Prepare input data
processor = WeatherDataProcessor()
df = processor.load_data("new_data.csv")
df = processor.preprocess_data(df, date_col='timestamp_column')

# Make prediction for a specific date
target_date = '2022-04-20 12:00:00'
X_input = processor.prepare_prediction_input(df, target_date, date_col='timestamp_column')
temperature = model.predict_temperature(X_input, processor, df.columns.tolist())

print(f"Predicted temperature for {target_date}: {temperature:.2f}°C")
```

## Data Format

The library expects weather data with the following characteristics:

- A datetime column (can be configured)
- A temperature column called 'Temperature (C)' (can be adjusted in the code)
- Numeric feature columns representing weather conditions

Text-based columns are automatically dropped during preprocessing.

## Example

See the `examples` directory for a complete usage example.

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- NumPy
- Pandas
- scikit-learn

## License

MIT License