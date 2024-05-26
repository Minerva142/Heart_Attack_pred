import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image
import pickle

with open('best_model_hypertension.pkl', 'rb') as file:
    hypertension_model = pickle.load(file)

with open('hypertension_scaler.pkl', 'rb') as file:
    hypertension_scaler = pickle.load(file)

with open('best_model_stroke.pkl', 'rb') as file:
    stroke_model = pickle.load(file)

with open('stroke_scaler.pkl', 'rb') as file:
    stroke_scaler = pickle.load(file)

with open('best_model_diabetes.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)

with open('diabetes_scaler.pkl', 'rb') as file:
    diabetes_scaler = pickle.load(file)

with open('best_model_heart.pkl', 'rb') as file:
    heart_model = pickle.load(file)

with open('best_scaler_heart.pkl', 'rb') as file:
    heart_scaler = pickle.load(file)

#Create header
st.write("""# Heart Attack Risk Predictor""")
st.write("""## How it works""")
st.write("view your predictions about your health condition based on your answers to the questions on the side panel.")

#image
st.write("""## Training Flow Diagram""")
image = Image.open('Train_diag.png')
st.image(image)

st.write("""## Prediction Flow Diagram""")
image = Image.open('Test_diag_v2.png')
st.image(image)


#links
st.write("""## Dataset links""")

st.write("https://www.kaggle.com/datasets/prosperchuks/health-dataset?select=diabetes_data.csv")
st.write("https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset")

# model types
st.write("""## Trained Model Types""")
st.write("Hypertension: DecisionTreeClassifier")
st.write("Stroke: RandomForestClassifier")
st.write("Diabetes: XGBClassifier")
st.write("Heart Attack: RandomForestClassifier")

#Bring in the data
data = pd.read_csv('heart_attack_prediction_dataset.csv')
st.write("## HEART ATTACK TRAIN DATA")
st.dataframe(data)

#Create and name sidebar
st.sidebar.header('Fill your survey')

st.sidebar.write("""#### Choose your values""")
def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 25, 1)
    sex = st.sidebar.slider('Sex (Male : 0, Female : 1)', 0, 1, 0, 1)
    gen_health = st.sidebar.slider('General Health scale 1 = excellent ,2 = very good, 3 = good, 4 = fair, 5 = poor', 1, 5, 3, 1)
    men_health = st.sidebar.slider('days of poor mental health scale 1-30 days', 0, 30, 0, 1)
    cholesterol = st.sidebar.slider('Cholesterol level', 0, 600, 150, 5)
    heart_rate = st.sidebar.slider('Heart Rate', 0, 160, 60, 1)
    family_history = st.sidebar.slider('Family History(for heart attack). 0 = no,1 = yes', 0, 1, 0, 1)
    obesity = st.sidebar.slider('Obesity. 0 = no,1 = yes', 0, 1, 0, 1)
    alcohol_consumption = st.sidebar.slider('Alcohol Consumption(regularly).0 = no,1 = yes', 0, 1, 0, 1)
    smoking_status = st.sidebar.slider('Smoking(regularly).0 = no,1 = yes', 0, 1, 0, 1)
    exercise_hours = st.sidebar.slider('Exercise Hours Per Week', 0, 50, 15, 1)
    stress_level = st.sidebar.slider('Stress Level', 0, 10, 3, 1)
    sedentary_hours = st.sidebar.slider('Sedentary Hours Per Day', 0.0, 12.0, 6.0, 0.5)
    income = st.sidebar.slider('Income', 0, 500000, 0, 1000)
    education_level = st.sidebar.slider('Education level 1-10', 1, 10, 6, 1)
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 20.0, 0.1)
    triglycerides = st.sidebar.slider('Triglycerides Level', 0, 1000, 350, 10)
    physical_days = st.sidebar.slider('Physical Activity Days Per Week', 0, 7, 3, 1)
    sleep_hours = st.sidebar.slider('Sleep Hours Per Day', 0.0, 16.0, 8.0, 0.5)
    systolic = st.sidebar.slider('Blood Pressure (Systolic)', 0, 200, 140, 1)
    diastolic = st.sidebar.slider('Blood Pressure (Diastolic)', 0, 120, 80, 1)
    diff_walk = st.sidebar.slider('Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes', 0, 1, 0, 1)
    fruits = st.sidebar.slider('Consume Fruit 1 or more times per day. 0 = no,1 = yes',  0, 1, 0, 1)
    veggies = st.sidebar.slider('Consume Vegetables 1 or more times per day. 0 = no ,1 = yes',  0, 1, 0, 1)
    married = st.sidebar.slider('Ever Married. 0 = no,1 = yes',  0, 1, 0, 1)
    work_type = st.sidebar.slider('patient job type: 0 - Never_worked, 1 - children, 2 - Govt_job, 3 - Self-employed, 4 - Private', 0, 4, 0, 1)
    avg_glucose_level = st.sidebar.slider('Avg. glucose level',  0, 300, 100, 5)
    cp = st.sidebar.slider('Chest pain type: 0: asymptomatic 1: typical angina 2: atypical angina 3: non-anginal pain', 0, 3, 0, 1)
    trestbps = st.sidebar.slider('Resting blood pressure', 50, 250, 120, 1)
    thalach = st.sidebar.slider('Maximum heart rate achieved', 50, 250, 120, 1)
    exang = st.sidebar.slider('Exercise induced angina. 0 = no,1 = yes', 0, 1, 0, 1)
    oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest.', 0.0, 10.0, 0.0, 0.1)
    slope = st.sidebar.slider('The slope of the peak exercise ST segment: 0: upsloping 1: flat 2: downsloping', 0, 2, 2, 1)
    ca = st.sidebar.slider('Number of major vessels (0–3) colored by flourosopy', 0, 5, 0, 1)
    thal = st.sidebar.slider('3: Normal; 6: Fixed defect; 7: Reversable defect', 0, 10, 2, 1)

    user_data_hypertension = {
        'cp' : cp,
        'trestbps' : trestbps,
        'chol' : cholesterol,
        'thalach' : thalach,
        'exang' : exang,
        'oldpeak' : oldpeak,
        'slope' : slope,
        'ca' : ca,
        'thal' : thal,
    }

    features_hypertension = pd.DataFrame(user_data_hypertension, index=[0])
    features_hypertension_scaled = hypertension_scaler.transform(features_hypertension)
    pred_hypertension = hypertension_model.predict(features_hypertension_scaled)

    user_data_stroke = {
        'age' : age,
        'hypertension' : pred_hypertension[0],
        'heart_disease' : 0,
        'ever_married' : married,
        'work_type' : work_type,
        'avg_glucose_level' : avg_glucose_level,
        'bmi' : bmi,
        'smoking_status' : smoking_status
    }

    features_stroke = pd.DataFrame(user_data_stroke, index=[0])
    features_stroke_scaled = stroke_scaler.transform(features_stroke)
    pred_stroke = stroke_model.predict(features_stroke_scaled)

    if physical_days > 2:
        PhysHlth = 1
    else:
        PhysHlth = 0

    if exercise_hours > 8:
        PhysActivity = 1
    else:
        PhysActivity = 0

    age_level = ((age -18) // 5 ) + 1
    income_level  = (income // 50000 ) + 1

    user_data_diabetes = {
        'HighBP': pred_hypertension[0],
        'BMI': bmi,
        'Stroke': pred_stroke[0],
        'PhysActivity': PhysActivity,
        'Fruits': fruits,
        'Veggies': veggies,
        'HvyAlcoholConsump': alcohol_consumption,
        'GenHlth': gen_health,
        'MentHlth': men_health,
        'PhysHlth': PhysHlth,
        'DiffWalk': diff_walk,
        'Sex': 1 - sex,
        'Age': age_level,
        'Education': education_level,
        'Income': income_level
    }

    features_diabetes = pd.DataFrame(user_data_diabetes, index=[0])
    features_diabetes_scaled = diabetes_scaler.transform(features_diabetes)
    pred_diabetes = diabetes_model.predict(features_diabetes_scaled)

    user_data_heart_attack ={
        'Age': age,
        'Cholesterol': cholesterol,
        'Heart Rate': heart_rate,
        'Diabetes': pred_diabetes[0],
        'Family History': family_history,
        'Obesity': obesity,
        'Alcohol Consumption': alcohol_consumption,
        'Exercise Hours Per Week' : exercise_hours,
        'Stress Level': stress_level,
        'Sedentary Hours Per Day': sedentary_hours,
        'Income': income,
        'BMI': bmi,
        'Triglycerides': triglycerides,
        'Physical Activity Days Per Week': physical_days,
        'Sleep Hours Per Day': sleep_hours,
        'BP_Systolic': systolic,
        'BP_Diastolic': diastolic,
        'Sex_Female': sex,
        'Sex_Male': 1 - sex,
    }

    features_heart_attack = pd.DataFrame(user_data_heart_attack, index=[0])
    features_heart_attack_scaled = heart_scaler.transform(features_heart_attack)
    pred_heart = heart_model.predict(features_heart_attack_scaled)

    return features_stroke,pred_stroke, features_hypertension, pred_hypertension, features_diabetes, pred_diabetes, features_heart_attack, pred_heart

df_stroke, pred_stroke,df_hypertension, pred_hypertension,df_diabetes, pred_diabetes, df_heart_attack, pred_heart = user_input_features()



st.write("## YOUR PREDICTIONS: ")

st.write("## Hypertension User Input: ")
df_hypertension
st.write("Predicted Hypertension: ")
pred_hypertension
st.write("## Stroke User Input and Hypertension(pred. vals added): ")
df_stroke
st.write("Predicted Stroke: ")
pred_stroke
st.write("## Diabetes User Input and Hypertension and Stroke(pred. vals added): ")
df_diabetes
st.write("Predicted Diabetes: ")
pred_diabetes
st.write("## Heart Attack User Input and Diabetes: ")
df_heart_attack
st.write("Predicted Heart Attack: ")
pred_heart




