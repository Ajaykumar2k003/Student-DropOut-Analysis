import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Student Dropout Analysis")

# Load dataset
@st.cache_data
def load_data():
    # Load your data here. Change the path to your actual dataset file
    data = pd.read_csv("D:\\Titan\\Projects\\Projects\\studentDropOutAnalysis\\dataset.csv")
    return data

# Load the data
data = load_data()
num = st.slider("Number of rows to view", 1, 100)

num = -1 if st.checkbox("Show all rows") else num

# Display dataset
st.write(data.head(num))

# Preprocessing data
st.subheader("Preprocessing")
# Display basic info about the dataset
if st.checkbox("Show Dataset Info"):
    st.write(data.info())

# Display statistical summary
if st.checkbox("Show Statistical Summary"):
    st.write(data.describe())

# Encode the target variable
label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

# Data Visualization
st.subheader("Data Visualization")

# Visualize dropout vs graduate counts
fig1, ax1 = plt.subplots()
sns.countplot(data=data, x='Target', ax=ax1)
st.pyplot(fig1)

# Visualize relationships with age at enrollment
fig2, ax2 = plt.subplots()
sns.boxplot(data=data, x='Target', y='Age at enrollment', ax=ax2)
st.pyplot(fig2)

# Model Training
st.subheader("Model Training")
st.write("Train a model to predict dropout rates.")

# Select features and target
X = data.drop(columns=['Target'])
y = data['Target']

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Display classification report
st.subheader("Model Evaluation")
st.text("Classification Report:")
st.text(classification_report(y_test, predictions))

# Display confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, predictions)
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax3, cmap='Blues', linewidths=1)
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')
ax3.set_xticklabels(label_encoder.classes_)
ax3.set_yticklabels(label_encoder.classes_)
st.pyplot(fig3)

# User input for prediction
st.subheader("Predict Dropout/Graduate/Enrolled")
input_data = {}

# Collect user inputs for each feature in the dataset
for column in X.columns:
    input_data[column] = st.number_input(f"Enter {column}", value=0)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)
    # Decode the predicted label back to the original category
    predicted_class = label_encoder.inverse_transform(prediction)
    st.write("Predicted Class:", predicted_class[0])

# Display additional insights
st.subheader("Additional Insights")
if st.checkbox("Show Correlation Heatmap"):
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=1, ax=ax4)
    plt.title('Correlation Heatmap')
    st.pyplot(fig4)

# Feature Importance
st.subheader("Feature Importance")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
fig5, ax5 = plt.subplots()
sns.barplot(x=importances[indices], y=X.columns[indices], ax=ax5)
ax5.set_title("Feature Importance")
ax5.set_xlabel("Relative Importance")
st.pyplot(fig5)

# Conclusion
st.subheader("Conclusion")
st.write("This application provides insights into student dropout rates based on various factors.")
