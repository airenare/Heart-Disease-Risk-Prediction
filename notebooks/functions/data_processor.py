import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../../data/interim/data_imputed_2.csv")


# Function to classify continuous column into 3 classes
def classify_col_by_3(data, cont_col, class_col, cutoff1, cutoff2):
    data[class_col] = 0
    data.loc[data[cont_col] < cutoff1, class_col] = 0
    data.loc[(data[cont_col] >= cutoff1) & (data[cont_col] < cutoff2), class_col] = 1
    data.loc[data[cont_col] >= cutoff2, class_col] = 2


# Function that classifies the cigsPerDay column into 4 classes
def classify_column(data, cont_col, class_col, cutoff_list):
    data[class_col] = 0
    for i, cutoff in enumerate(cutoff_list):
        data.loc[data[cont_col] > cutoff, class_col] = i + 1


# Function to calculate the MAP (mean arterial pressure) from the systolic and diastolic blood pressure
def calculate_map(data, sys_col, dia_col, map_col):
    data[map_col] = ((data[sys_col] - data[dia_col]) / 3 + data[dia_col]).apply(
        lambda x: round(x, 2)
    )


# Education column
df["education"] = df["education"] - 1

# Call function over the glucose column
classify_col_by_3(
    data=df, cont_col="glucose", class_col="diabetes_stage", cutoff1=100, cutoff2=126
)

# Call function over the cigsPerDay column
classify_column(
    data=df, cont_col="cigsPerDay", class_col="smoker_class", cutoff_list=[0, 10, 20]
)

# Call function over the sysBP and diaBP columns
calculate_map(data=df, sys_col="sysBP", dia_col="diaBP", map_col="MAP")

map_htn_cutoffs = [105.67, 119.0, 132.33]

# Call classify_column function over the MAP column
classify_column(
    data=df, cont_col="MAP", class_col="hypertension_stage", cutoff_list=map_htn_cutoffs
)

# Columns to drop
cols_to_drop = [
    "cigsPerDay",
    "currentSmoker",
    "diabetes",
    "glucose",
    "sysBP",
    "diaBP",
    "prevalentHyp",
]
# Drop the columns
df = df.drop(cols_to_drop, axis=1)

# Get dummy variables for education, diabetes_stage, smoker_class, hypertension_stage
df = pd.get_dummies(
    df,
    columns=["education", "diabetes_stage", "smoker_class", "hypertension_stage"],
    drop_first=True,
    dtype=int,
)

# Change the order of the columns
categorical_columns = [
    "male",
    "BPMeds",
    "prevalentStroke",
    "education_1.0",
    "education_2.0",
    "education_3.0",
    "diabetes_stage_1",
    "diabetes_stage_2",
    "smoker_class_1",
    "smoker_class_2",
    "smoker_class_3",
    "hypertension_stage_1",
    "hypertension_stage_2",
    "hypertension_stage_3",
]

continuous_columns = ["age", "totChol", "BMI", "heartRate", "MAP"]

target = ["TenYearCHD"]

df = df[categorical_columns + continuous_columns + target]

# Scale the continuous columns
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[continuous_columns])
scaled_df = pd.DataFrame(scaled_df, columns=continuous_columns)

df[continuous_columns] = scaled_df

# Display the first few rows of the dataset
df.head()

# Ask the user if they want to save the processed data
save = input("Do you want to save the processed data? (y/n): ")

if save == "y":
    # Ask the user for the file name
    file_name = input("Enter the file name: ")
    # Save the processed data
    df.to_csv("../../data/processed/file_name", index=False)
    print("Data saved successfully!")

else:
    print("Data not saved!")
