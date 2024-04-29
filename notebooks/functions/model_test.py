import pandas as pd

# Import the logistic regression model
from sklearn.linear_model import LogisticRegression
from joblib import load
from sklearn.preprocessing import StandardScaler

from process_data import process_data

# Load the model
model = load("../../models/logistic_regression_balanced_best.pkl")

# Load the data
df = pd.read_csv("../../data/raw/Framingham Heart Disease.csv").dropna()


# X = df.drop('TenYearCHD', axis=1)

# Create a StandardScaler object
scaler = StandardScaler()
scaler.fit(df)

# Define the input data
male = 1
BPMeds = 0
prevalentStroke = 0
education_1 = 0
education_2 = 0
education_3 = 1
diabetes_stage_1 = 0
diabetes_stage_2 = 0
smoker_class_1 = 1
smoker_class_2 = 0
smoker_class_3 = 0
hypertension_stage_1 = 0
hypertension_stage_2 = 0
hypertension_stage_3 = 0
age = 32
totChol = 200
BMI = 22
heartRate = 80
MAP = 80 + (120 - 80) / 3


# Create a dictionary of the input data
input_dict = {
    "male": male,
    "BPMeds": BPMeds,
    "prevalentStroke": prevalentStroke,
    "education_1.0": education_1,
    "education_2.0": education_2,
    "education_3.0": education_3,
    "diabetes_stage_1": diabetes_stage_1,
    "diabetes_stage_2": diabetes_stage_2,
    "smoker_class_1": smoker_class_1,
    "smoker_class_2": smoker_class_2,
    "smoker_class_3": smoker_class_3,
    "hypertension_stage_1": hypertension_stage_1,
    "hypertension_stage_2": hypertension_stage_2,
    "hypertension_stage_3": hypertension_stage_3,
    "age": age,
    "totChol": totChol,
    "BMI": BMI,
    "heartRate": heartRate,
    "MAP": MAP,
    "TenYearCHD": 0,
}

# Create a DataFrame from the dictionary
input_df = pd.DataFrame(input_dict, index=[9999])

# Set the datatypes for the input data
dtypes = {
    "male": "int64",
    "BPMeds": "float64",
    "prevalentStroke": "int64",
    "education_1.0": "int64",
    "education_2.0": "int64",
    "education_3.0": "int64",
    "diabetes_stage_1": "int64",
    "diabetes_stage_2": "int64",
    "smoker_class_1": "int64",
    "smoker_class_2": "int64",
    "smoker_class_3": "int64",
    "hypertension_stage_1": "int64",
    "hypertension_stage_2": "int64",
    "hypertension_stage_3": "int64",
    "age": "float64",
    "totChol": "float64",
    "BMI": "float64",
    "heartRate": "float64",
    "MAP": "float64",
    "TenYearCHD": "int64",
}

# Convert the input data to the correct datatypes
input_df = input_df.astype(dtypes)

# Process the data
processed_df = process_data(df, added_df=input_df, save=False)

print(processed_df)
