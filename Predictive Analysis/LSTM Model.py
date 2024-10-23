import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Read and prepare data
data = pd.DataFrame([
    [2005, 5359.199997], [2006, 4127.900055], [2007, 9561],
    [2008, 7032.299988], [2009, 5161], [2010, 9262.000092],
    [2011, 11774.49951], [2012, 12124.2002], [2013, 13246.89981],
    [2014, 12366.10028], [2015, 9293.600037], [2016, 12015.50003],
    [2017, 12489.79999], [2018, 11548.90024], [2019, 8900.899811],
    [2020, 8603.900192], [2021, 13544.60007], [2022, 13406.50005]
], columns=['year', 'oil_sector'])

# Enhanced feature engineering
def create_features(df):
    df = df.copy()
    
    # Time-based features
    df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    
    # Technical indicators
    df['SMA_3'] = df['oil_sector'].rolling(window=3).mean()
    df['SMA_5'] = df['oil_sector'].rolling(window=5).mean()
    df['momentum'] = df['oil_sector'].diff(1)
    df['acceleration'] = df['momentum'].diff(1)
    
    # Volatility features
    df['rolling_std_3'] = df['oil_sector'].rolling(window=3).std()
    df['rolling_std_5'] = df['oil_sector'].rolling(window=5).std()
    
    # Cyclical features
    df['cycle_3'] = df['oil_sector'].rolling(window=3).mean() - df['oil_sector']
    df['cycle_5'] = df['oil_sector'].rolling(window=5).mean() - df['oil_sector']
    
    # Growth rates
    df['growth_rate'] = df['oil_sector'].pct_change()
    df['growth_rate_sma'] = df['growth_rate'].rolling(window=3).mean()
    
    # Fill NaN values with appropriate methods
    for col in df.columns:
        if col != 'year':
            df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
    
    return df

# Apply feature engineering
enhanced_data = create_features(data)

# Select features for modeling
feature_columns = [
    'oil_sector', 'year_norm', 'SMA_3', 'SMA_5', 'momentum', 'acceleration',
    'rolling_std_3', 'rolling_std_5', 'cycle_3', 'cycle_5', 'growth_rate', 'growth_rate_sma'
]

# Scale features using RobustScaler
scaler = RobustScaler()
scaled_values = scaler.fit_transform(enhanced_data[feature_columns])

# Create sequences with dynamic adjustment of importance
def create_weighted_sequences(values, seq_length):
    X, y = [], []
    for i in range(len(values) - seq_length):
        sequence = values[i:(i + seq_length)]
        X.append(sequence)
        y.append(values[i + seq_length, 0])  # Target is oil_sector
    return np.array(X), np.array(y)

# Parameters
sequence_length = 3
n_features = len(feature_columns)

# Split data
train_data = scaled_values[:-2]
test_data = scaled_values[-2:]

# Create sequences
X_train, y_train = create_weighted_sequences(train_data, sequence_length)

# Build improved model
def build_model(sequence_length, n_features):
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, 
             input_shape=(sequence_length, n_features),
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
             recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.2),
        LSTM(64, activation='tanh',
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
             recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dense(1, activation='linear')
    ])
    return model

model = build_model(sequence_length, n_features)

# Use a fixed learning rate instead of a schedule
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='huber')

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=50,
    restore_best_weights=True,
    min_delta=1e-4
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=20,
    min_lr=1e-6
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=4,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

# Rest of the code remains the same...
def predict_sequence(model, last_sequence, n_steps, scaler, data):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for i in range(n_steps):
        current_input = current_sequence.reshape((1, sequence_length, n_features))
        next_pred = model.predict(current_input, verbose=0)[0][0]
        predictions.append(next_pred)
        
        if i < n_steps - 1:
            new_features = create_features(pd.concat([
                data,
                pd.DataFrame({'year': [data['year'].iloc[-1] + i + 1], 
                            'oil_sector': [scaler.inverse_transform([[next_pred] + [0]*(n_features-1)])[0][0]]})
            ])).iloc[-1]
            
            new_row = scaler.transform(new_features[feature_columns].values.reshape(1, -1))[0]
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row
    
    return np.array(predictions)

# Make predictions
last_sequence = scaled_values[-sequence_length-2:-2].reshape(sequence_length, n_features)
test_predictions = predict_sequence(model, last_sequence, 2, scaler, enhanced_data)
future_sequence = scaled_values[-sequence_length:].reshape(sequence_length, n_features)
future_predictions = predict_sequence(model, future_sequence, 3, scaler, enhanced_data)

# Inverse transform predictions
test_predictions_inv = scaler.inverse_transform(
    np.column_stack([test_predictions, np.zeros((len(test_predictions), n_features-1))])
)[:, 0]
future_predictions_inv = scaler.inverse_transform(
    np.column_stack([future_predictions, np.zeros((len(future_predictions), n_features-1))])
)[:, 0]

# Print results
actual_test = data['oil_sector'][-2:].values
print("\nModel Performance Metrics (on test set 2021-2022):")
print(f"Mean Squared Error: {mean_squared_error(actual_test, test_predictions_inv):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(actual_test, test_predictions_inv)):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(actual_test, test_predictions_inv):.2f}")
print(f"R² Score: {r2_score(actual_test, test_predictions_inv):.2f}")

print("\nPredicted values for 2023-2025:")
for year, pred in zip([2023, 2024, 2025], future_predictions_inv):
    print(f"{year}: {pred:.2f}")

print("\nComparison of Actual vs Predicted for Test Years:")
for year, actual, pred in zip(data['year'][-2:], actual_test, test_predictions_inv):
    print(f"{year} - Actual: {actual:.2f}, Predicted: {pred:.2f}, Difference: {actual - pred:.2f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(data['year'], data['oil_sector'], label='Historical Data', marker='o')
plt.plot(data['year'][-2:], test_predictions_inv, label='Test Predictions', marker='s', color='red')
plt.plot([2023, 2024, 2025], future_predictions_inv, label='Future Predictions', marker='d', linestyle='--', color='purple')
plt.title('Oil Sector: Enhanced LSTM Model Predictions')
plt.xlabel('Year')
plt.ylabel('Oil Sector Value')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

for year, value in zip(data['year'][-2:], test_predictions_inv):
    plt.annotate(f'{value:.2f}', (year, value), textcoords="offset points", xytext=(0,10), ha='center')
for year, value in zip([2023, 2024, 2025], future_predictions_inv):
    plt.annotate(f'{value:.2f}', (year, value), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()