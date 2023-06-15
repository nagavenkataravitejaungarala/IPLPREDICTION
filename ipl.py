
pip install pandas
pip install scikit-learn
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title of the application
st.title("IPL Match Prediction")

# Upload the CSV file
csv_file = st.file_uploader("Upload CSV", type="csv")

if csv_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Display the DataFrame
    st.dataframe(df)

    # Split the data into features and target variable
    X = df.drop('outcome', axis=1)
    y = df['outcome']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    classifier = RandomForestClassifier()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Display the accuracy
    st.write(f"Accuracy: {accuracy}")
