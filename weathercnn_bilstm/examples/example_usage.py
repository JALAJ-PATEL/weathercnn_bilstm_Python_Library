"""
Example usage of the WeatherCNN_BiLSTM library

This script demonstrates how to use the library to:
1. Load and preprocess weather data
2. Create a CNN-BiLSTM model
3. Train and evaluate the model
4. Make predictions
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the library
from weathercnn_bilstm import WeatherPredictor, WeatherDataProcessor

def main():
    """Main function to demonstrate the library usage"""
    print("👋 Welcome to the WeatherCNN_BiLSTM example!")
    
    # 1. Data Loading and Preprocessing
    print("\n📊 Loading and preprocessing data...")
    data_path = "../Weather-Data.csv"  # Adjust path as needed
    
    # Initialize data processor
    processor = WeatherDataProcessor(sequence_length=24)
    
    # Load and preprocess data
    df = processor.load_data(data_path)
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Preprocess the data - critical to drop any datetime columns
    df = processor.preprocess_data(df, date_col='Formatted Date')
    print(f"Data preprocessed: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Data columns after preprocessing:", df.columns.tolist())
    
    # 2. Create sequences
    print("\n🔄 Creating sequences...")
    X, y, feature_cols = processor.create_sequences(df)
    print(f"Sequences created: X shape: {X.shape}, y shape: {y.shape}")
    print("Feature columns used:", feature_cols)
    
    # 3. Split data into train, validation, test sets
    print("\n✂️ Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = processor.train_val_test_split(
        X, y, test_size=0.15, val_size=0.15, shuffle=False
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 4. Build and train the model
    print("\n🧠 Building and training model...")
    model = WeatherPredictor()
    model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = model.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=64, verbose=1)  # Reduced epochs for example
    
    # 5. Evaluate the model
    print("\n📏 Evaluating model...")
    results = model.evaluate(X_test, y_test, data_processor=processor, feature_cols=feature_cols)
    print(f"Test MSE: {results['mse']:.4f}")
    print(f"Test RMSE: {results['rmse']:.4f}")
    print(f"Test MAE: {results['mae']:.4f}")
    
    # 6. Plot some predictions (only a subset for visualization)
    print("\n📈 Plotting predictions...")
    plt.figure(figsize=(12, 5))
    plt.plot(results['y_true'][:100], label="Actual Temperature")
    plt.plot(results['y_pred'][:100], label="Predicted Temperature")
    plt.title("Actual vs Predicted Temperature")
    plt.xlabel("Time Steps (Hours)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("temperature_prediction.png")
    print("Plot saved as 'temperature_prediction.png'")
    
    # 7. Save the model
    print("\n💾 Saving model...")
    model.save_model("weather_model.keras")
    
    # 8. Load the model and make a single prediction
    print("\n🔮 Loading model and making a prediction...")
    loaded_model = WeatherPredictor.load_from_file("weather_model.keras")
    
    # Get a sample from the test set
    sample_X = X_test[0:1]
    
    # Make prediction
    prediction = loaded_model.predict_temperature(
        sample_X, data_processor=processor, feature_cols=feature_cols
    )
    
    actual = processor.inverse_transform_temp(y_test[0:1], feature_cols)[0]
    print(f"Actual temperature: {actual:.2f}°C")
    print(f"Predicted temperature: {prediction:.2f}°C")
    
    print("\n✅ Example completed!")

if __name__ == "__main__":
    main()