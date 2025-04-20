"""
CNN-BiLSTM model for weather temperature prediction
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error


class WeatherPredictor:
    """
    CNN-BiLSTM model for predicting weather temperature.
    
    This class builds, trains, and uses a CNN-BiLSTM neural network for
    temperature forecasting based on historical weather data.
    """
    
    def __init__(self, input_shape=None):
        """
        Initialize the weather predictor.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
        """
        self.model = None
        self.input_shape = input_shape
        self.history = None
        self.feature_cols = None
    
    def build_model(self, input_shape=None):
        """
        Build the CNN-BiLSTM model architecture.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            
        Returns:
            The built model
        """
        if input_shape:
            self.input_shape = input_shape
            
        if not self.input_shape:
            raise ValueError("Input shape must be provided")
            
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=64, patience=5, verbose=1):
        """
        Train the CNN-BiLSTM model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            X_val: Validation input sequences
            y_val: Validation target values
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if not self.model:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Scaled predictions
        """
        if not self.model:
            raise ValueError("Model not built yet. Call build_model() and train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, data_processor=None, feature_cols=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test input sequences
            y_test: Test target values
            data_processor: WeatherDataProcessor instance for inverse scaling
            feature_cols: Feature column names
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model:
            raise ValueError("Model not built yet")
            
        # Make predictions
        y_pred_scaled = self.model.predict(X_test)
        
        # If a data processor is provided, inverse transform the values
        if data_processor and feature_cols:
            y_pred_actual = data_processor.inverse_transform_temp(y_pred_scaled, feature_cols)
            
            # Extract true values and inverse transform
            temp_index = feature_cols.index(data_processor.target_col)
            scaled_actual = np.zeros((len(y_test), 1))
            scaled_actual[:, 0] = y_test
            y_test_actual = data_processor.inverse_transform_temp(scaled_actual, feature_cols)
        else:
            y_pred_actual = y_pred_scaled
            y_test_actual = y_test
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'y_true': y_test_actual,
            'y_pred': y_pred_actual
        }
    
    def save_model(self, filepath='weather_temp_predictor.keras'):
        """
        Save the model to a file.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.model:
            raise ValueError("No model to save. Call build_model() and train() first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath='weather_temp_predictor.keras'):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            WeatherPredictor instance with loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        predictor = cls()
        predictor.model = load_model(filepath)
        predictor.input_shape = predictor.model.input_shape[1:]
        
        return predictor
    
    def predict_temperature(self, X_input, data_processor=None, feature_cols=None):
        """
        Predict temperature and inverse transform if possible.
        
        Args:
            X_input: Input data for prediction
            data_processor: WeatherDataProcessor instance for inverse scaling
            feature_cols: Feature column names
            
        Returns:
            Predicted temperature(s)
        """
        y_pred_scaled = self.predict(X_input)
        
        # If a data processor is provided, inverse transform the values
        if data_processor and feature_cols:
            return data_processor.inverse_transform_temp(y_pred_scaled, feature_cols)[0]
        
        return y_pred_scaled[0][0]