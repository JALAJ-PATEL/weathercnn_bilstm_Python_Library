"""
Data processing utilities for weather prediction with CNN-BiLSTM
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class WeatherDataProcessor:
    """
    Handles data preprocessing for the Weather CNN-BiLSTM model.
    
    This class manages loading, cleaning, scaling, and sequence creation from weather datasets.
    """
    
    def __init__(self, sequence_length=24):
        """
        Initialize the data processor.
        
        Args:
            sequence_length: Number of time steps to use for sequence data (default: 24 hours)
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.target_col = 'Temperature (C)'
        self._is_fitted = False
    
    def load_data(self, file_path, date_col='Formatted Date'):
        """
        Load weather data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            date_col: Name of the datetime column
            
        Returns:
            DataFrame with the loaded data
        """
        df = pd.read_csv(file_path)
        
        # Check if the dataframe has the required column
        if self.target_col not in df.columns:
            raise ValueError(f"Dataset must contain '{self.target_col}' column")
        
        return df
    
    def preprocess_data(self, df, date_col=None):
        """
        Preprocess the weather data by extracting datetime components and handling missing values.
        
        Args:
            df: Weather DataFrame
            date_col: Name of the datetime column, if any
            
        Returns:
            Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Extract datetime features if date column exists
        if date_col and date_col in df_copy.columns:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], utc=True)
            df_copy['Year'] = df_copy[date_col].dt.year
            df_copy['Month'] = df_copy[date_col].dt.month
            df_copy['Day'] = df_copy[date_col].dt.day
            df_copy['Hour'] = df_copy[date_col].dt.hour
            
            # Drop the original date column since we've extracted its components
            df_copy.drop(columns=[date_col], inplace=True)
        
        # Drop any text columns or datetime columns (cannot be used by the model)
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy.drop(columns=[col], inplace=True)
        
        # Handle missing values
        df_copy.dropna(inplace=True)
        
        return df_copy
    
    def create_sequences(self, df):
        """
        Create sequences for time series modeling.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            X: Input sequences
            y: Target values
            feature_cols: List of feature column names
        """
        # Make sure we have only numeric data before scaling
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] < df.shape[1]:
            print(f"Warning: Dropped {df.shape[1] - numeric_df.shape[1]} non-numeric columns before scaling")
        
        # Get feature column names after filtering for numeric columns only
        feature_cols = numeric_df.columns.tolist()
        
        if not self._is_fitted:
            # Fit the scaler on the numeric data only
            self.scaler.fit(numeric_df)
            self._is_fitted = True
        
        # Transform the data
        scaled_data = self.scaler.transform(numeric_df)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i][feature_cols.index(self.target_col)])
        
        X, y = np.array(X), np.array(y)
        
        return X, y, feature_cols
    
    def train_val_test_split(self, X, y, test_size=0.15, val_size=0.15, shuffle=False):
        """
        Split the data into train, validation, and test sets.
        
        Args:
            X: Input sequences
            y: Target values
            test_size: Proportion of data to use for testing
            val_size: Proportion of remaining data to use for validation
            shuffle: Whether to shuffle the data before splitting
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split off the test set
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        
        # Then split the remaining data into train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for the reduced size
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, shuffle=shuffle)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def inverse_transform_temp(self, scaled_values, feature_cols):
        """
        Inverse transform scaled temperature values back to original scale.
        
        Args:
            scaled_values: Scaled temperature predictions or actual values
            feature_cols: List of feature column names
            
        Returns:
            Temperature values in original scale
        """
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        temp_index = feature_cols.index(self.target_col)
        
        # Create a temporary scaler for just the temperature column
        temp_scaler = MinMaxScaler()
        temp_scaler.min_ = np.array([self.scaler.min_[temp_index]])
        temp_scaler.scale_ = np.array([self.scaler.scale_[temp_index]])
        temp_scaler.data_min_ = np.array([self.scaler.data_min_[temp_index]])
        temp_scaler.data_max_ = np.array([self.scaler.data_max_[temp_index]])
        
        # Reshape for inverse transform
        scaled_values_reshaped = np.array(scaled_values).reshape(-1, 1)
        return temp_scaler.inverse_transform(scaled_values_reshaped).flatten()
    
    def prepare_prediction_input(self, df, target_date, date_col=None):
        """
        Prepare input sequence for prediction for a specific target date.
        
        Args:
            df: Weather DataFrame
            target_date: Target datetime for prediction
            date_col: Name of the datetime column
            
        Returns:
            X_input: Input sequence for prediction
        """
        if not date_col or date_col not in df.columns:
            raise ValueError("Date column is required for prediction input preparation")
        
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Filter data for the previous sequence_length hours before target_date
        target_date = pd.to_datetime(target_date)
        start_time = target_date - pd.Timedelta(hours=self.sequence_length)
        
        mask = (df_copy[date_col] >= start_time) & (df_copy[date_col] < target_date)
        input_data = df_copy.loc[mask]
        
        if len(input_data) < self.sequence_length:
            raise ValueError(f"Not enough historical data available (need {self.sequence_length} hours)")
            
        # Drop the date column if it exists to match training format
        if date_col in input_data.columns:
            input_data.drop(columns=[date_col], inplace=True)
        
        # Scale the input data
        scaled_input = self.scaler.transform(input_data.tail(self.sequence_length))
        
        # Reshape for model input
        X_input = np.expand_dims(scaled_input, axis=0)  # Shape: (1, sequence_length, features)
        
        return X_input