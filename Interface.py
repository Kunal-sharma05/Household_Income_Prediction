import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Define function to predict Total Household Income based on user input
def predict_income(features):
    # Load the saved model
    with open('My_Linear_Model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Create input array for the model
    input_array = np.array([features])

    # Make prediction using the loaded model
    income_prediction = model.predict(input_array)[0]

    return income_prediction

# Define Streamlit app
def main():
    st.title("Total Household Income Prediction")

    # Add input fields for user to enter data
    communication_expenditure = st.number_input("Communication Expenditure")
    housing_expenditure = st.number_input("Housing and Water Expenditure")
    miscellaneous_expenditure = st.number_input("Miscellaneous Goods and Services Expenditure")
    food_expenditure = st.number_input("Total Food Expenditure")
    transportation_expenditure = st.number_input("Transportation Expenditure")
    clothing_expenditure = st.number_input("Clothing, Footwear and Other Wear Expenditure")
    rental_value = st.number_input("Imputed House Rental Value")
    meat_expenditure = st.number_input("Meat Expenditure")
    entrepreneurial_income = st.number_input("Total Income from Entrepreneurial Activities")
    computer_count = st.number_input("Number of Personal Computers")

    # Add button to submit input and get prediction
    if st.button("Predict Income"):
        # Create feature array from user inputs
        features = [communication_expenditure, housing_expenditure, miscellaneous_expenditure, food_expenditure,
                    transportation_expenditure, clothing_expenditure, rental_value, meat_expenditure,
                    entrepreneurial_income, computer_count]

        # Call predict_income function to get prediction
        income_prediction = predict_income(features)

        # Display the predicted income
        st.write("Predicted Total Household Income: ", round(income_prediction, 2))

if __name__ == '__main__':
    main()
