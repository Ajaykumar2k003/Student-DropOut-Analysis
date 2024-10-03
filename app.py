import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and label encoder
model = joblib.load('D:\Titan\Git\StudentDropOutAnalysis\graduate_dropout_enrolled_model.pkl')
label_encoder = joblib.load('D:\Titan\Git\StudentDropOutAnalysis\label_encoder.pkl')

# Load the dataset to get default values
data = pd.read_csv('D:\Titan\Git\StudentDropOutAnalysis\dataset.csv')

# Debug: Print column names to verify them
st.write("Column names in the dataset:")
st.write(data.columns.tolist())

# Select a sample row (you can change the index based on your dataset)
sample_row = data.sample(1).iloc[0]

# Function to preprocess and predict the outcome based on user input
def predict_outcome(user_input):
    prediction = model.predict(np.array(user_input).reshape(1, -1))
    return label_encoder.inverse_transform(prediction)[0]  # Convert back to 'Dropout', 'Graduate', or 'Enrolled'

# Streamlit App UI
st.title('Student Outcome Prediction')
st.write("Enter the student's details to predict if they are 'Dropout', 'Graduate', or 'Enrolled'.")

# Input fields based on the dataset columns with default values from the sample row
marital_status = st.number_input('Marital status', min_value=0, max_value=5, value=int(sample_row['marital_status']))
application_mode = st.number_input('Application mode', min_value=1, max_value=20, value=int(sample_row['application_mode']))
application_order = st.number_input('Application order', min_value=1, max_value=10, value=int(sample_row['application_order']))
course = st.number_input('Course', min_value=1, max_value=170, value=int(sample_row['course']))
attendance = st.selectbox('Daytime/Evening attendance', [0, 1], index=int(sample_row['attendance']))
previous_qualification = st.number_input('Previous qualification', min_value=1, max_value=10, value=int(sample_row['previous_qualification']))
nationality = st.number_input('Nationality', min_value=1, max_value=244, value=int(sample_row['nationality']))
mother_qualification = st.number_input('Mother\'s qualification', min_value=1, max_value=6, value=int(sample_row['mother_qualification']))
father_qualification = st.number_input('Father\'s qualification', min_value=1, max_value=6, value=int(sample_row['father_qualification']))
mother_occupation = st.number_input('Mother\'s occupation', min_value=0, max_value=5, value=int(sample_row['mother_occupation']))
father_occupation = st.number_input('Father\'s occupation', min_value=0, max_value=5, value=int(sample_row['father_occupation']))
displaced = st.selectbox('Displaced', [0, 1], index=int(sample_row['displaced']))
special_needs = st.selectbox('Educational special needs', [0, 1], index=int(sample_row['special_needs']))
debtor = st.selectbox('Debtor', [0, 1], index=int(sample_row['debtor']))
tuition_up_to_date = st.selectbox('Tuition fees up to date', [0, 1], index=int(sample_row['tuition_up_to_date']))
gender = st.selectbox('Gender', [0, 1], index=int(sample_row['gender']))  # 0 for Female, 1 for Male
scholarship = st.selectbox('Scholarship holder', [0, 1], index=int(sample_row['scholarship']))
age_enrollment = st.number_input('Age at enrollment', min_value=17, max_value=100, value=int(sample_row['age_enrollment']))
international = st.selectbox('International student', [0, 1], index=int(sample_row['international']))

# Curricular units - 1st semester
units_1st_sem_credited = st.number_input('Curricular units 1st sem (credited)', min_value=0, max_value=10, value=int(sample_row['units_1st_sem_credited']))
units_1st_sem_enrolled = st.number_input('Curricular units 1st sem (enrolled)', min_value=0, max_value=10, value=int(sample_row['units_1st_sem_enrolled']))
units_1st_sem_evaluations = st.number_input('Curricular units 1st sem (evaluations)', min_value=0, max_value=10, value=int(sample_row['units_1st_sem_evaluations']))
units_1st_sem_approved = st.number_input('Curricular units 1st sem (approved)', min_value=0, max_value=10, value=int(sample_row['units_1st_sem_approved']))
units_1st_sem_grade = st.number_input('Curricular units 1st sem (grade)', min_value=0.0, max_value=20.0, value=float(sample_row['units_1st_sem_grade']))

# Curricular units - 2nd semester
units_2nd_sem_credited = st.number_input('Curricular units 2nd sem (credited)', min_value=0, max_value=10, value=int(sample_row['units_2nd_sem_credited']))
units_2nd_sem_enrolled = st.number_input('Curricular units 2nd sem (enrolled)', min_value=0, max_value=10, value=int(sample_row['units_2nd_sem_enrolled']))
units_2nd_sem_evaluations = st.number_input('Curricular units 2nd sem (evaluations)', min_value=0, max_value=10, value=int(sample_row['units_2nd_sem_evaluations']))
units_2nd_sem_approved = st.number_input('Curricular units 2nd sem (approved)', min_value=0, max_value=10, value=int(sample_row['units_2nd_sem_approved']))
units_2nd_sem_grade = st.number_input('Curricular units 2nd sem (grade)', min_value=0.0, max_value=20.0, value=float(sample_row['units_2nd_sem_grade']))

# Economic factors
unemployment_rate = st.number_input('Unemployment rate', min_value=0.0, max_value=50.0, value=float(sample_row['unemployment_rate']))
inflation_rate = st.number_input('Inflation rate', min_value=-10.0, max_value=50.0, value=float(sample_row['inflation_rate']))
gdp = st.number_input('GDP', min_value=-10.0, max_value=10.0, value=float(sample_row['gdp']))

# Gather all inputs into a list
user_input = [
    marital_status, application_mode, application_order, course, attendance, previous_qualification, nationality,
    mother_qualification, father_qualification, mother_occupation, father_occupation, displaced, special_needs, 
    debtor, tuition_up_to_date, gender, scholarship, age_enrollment, international, units_1st_sem_credited, 
    units_1st_sem_enrolled, units_1st_sem_evaluations, units_1st_sem_approved, units_1st_sem_grade,
    units_2nd_sem_credited, units_2nd_sem_enrolled, units_2nd_sem_evaluations, units_2nd_sem_approved, 
    units_2nd_sem_grade, unemployment_rate, inflation_rate, gdp
]

# Predict button
if st.button('Predict'):
    user_input_array = np.array(user_input).reshape(1, -1)  # Reshape for model input
    st.write(f'Input shape: {user_input_array.shape}')  # Check the shape here
    outcome = predict_outcome(user_input_array)
    st.success(f'The predicted outcome is: {outcome}')
