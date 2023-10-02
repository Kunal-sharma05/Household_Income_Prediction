import pandas as pd
import numpy as np
import tensorflow as tf 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error,r2_score 
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess the dataset
df = pd.read_csv('Family Income and Expenditure.csv')
features = ['Total Food Expenditure', 'Agricultural Household indicator', 'Bread and Cereals Expenditure', 'Total Rice Expenditure', 'Meat Expenditure', 'Total Fish and  marine products Expenditure', 'Fruit Expenditure', 'Vegetables Expenditure', 'Restaurant and hotels Expenditure', 'Alcoholic Beverages Expenditure', 'Tobacco Expenditure', 'Clothing, Footwear and Other Wear Expenditure', 'Housing and water Expenditure', 'Imputed House Rental Value', 'Medical Care Expenditure', 'Transportation Expenditure', 'Communication Expenditure', 'Education Expenditure', 'Miscellaneous Goods and Services Expenditure', 'Special Occasions Expenditure', 'Crop Farming and Gardening expenses', 'Total Income from Entrepreneurial Acitivites', 'Household Head Age', 'Total Number of Family members', 'Members with age less than 5 year old', 'Members with age 5 - 17 years old', 'Total number of family members employed', 'House Floor Area', 'House Age', 'Number of bedrooms', 'Electricity', 'Number of Television', 'Number of CD/VCD/DVD', 'Number of Component/Stereo set', 'Number of Refrigerator/Freezer', 'Number of Washing Machine', 'Number of Airconditioner', 'Number of Car, Jeep, Van', 'Number of Landline/wireless telephones', 'Number of Cellular phone', 'Number of Personal Computer', 'Number of Stove with Oven/Gas Range', 'Number of Motorized Banca', 'Number of Motorcycle/Tricycle']

target = 'Total Household Income'

X = df[features].values
y = df[target].values

scaler = MinMaxScaler()  # Scale features to the range [0, 1]
X = scaler.fit_transform(X)

# Use RFE with a Random Forest Regressor to select the most important features
estimator = RandomForestRegressor()
selector = RFE(estimator, n_features_to_select=15)
selector = selector.fit(X, y)
selected_features = [features[i] for i in range(len(features)) if selector.support_[i]]

# Update X to only include the selected features
X = df[selected_features].values
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
def create_model(optimizer='adam', activation='relu', units=64, dropout_rate=0.2):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units, activation=activation, input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(units, activation=activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(units, activation=activation),  # Additional hidden layer
        tf.keras.layers.Dropout(dropout_rate),
         tf.keras.layers.Dense(units, activation=activation),  # Additional hidden layer
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Train the model
model = create_model(optimizer='adam', activation='relu', units=64, dropout_rate=0.2)
model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)

# Make predictions on the test set with the final model
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('mse: ',mse)
print('r2_score: ',r2)

accuracy = r2 * 100
accuracy = round(accuracy, 2)
print('Accuracy:', accuracy, '%')

print(selected_features)
