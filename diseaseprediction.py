from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os

# Load and prepare the dataset
data = pd.read_csv(os.path.join("templates", "Training.csv"))
df = pd.DataFrame(data)

# Define features and target variable
cols = df.columns[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print("RandomForest - Ensemble Learning")

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
rf_accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
print(f"Random Forest Accuracy: {rf_accuracy:.2f}%")

# Gradient Boosting Classifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(x_train, y_train)
gb_accuracy = accuracy_score(y_test, gb.predict(x_test)) * 100
print(f"Gradient Boosting Accuracy: {gb_accuracy:.2f}%")

# Assign indices to symptoms
indices = [i for i in range(len(cols))]
symptoms = df.columns.values[:-1]
dictionary = dict(zip(symptoms, indices))

# Function to predict based on user input symptoms
def dosomething(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for _ in range(len(symptoms))]
    for i in user_input_symptoms:
        if i in dictionary:  # Check if symptom is valid
            idx = dictionary[i]
            user_input_label[idx] = 1

    user_input_label = np.array(user_input_label).reshape(1, -1)  # Reshape for a single sample
    rf_prediction = rf.predict(user_input_label)
    gb_prediction = gb.predict(user_input_label)
    return rf_prediction

# Example usage of dosomething function
# print(dosomething(['headache', 'muscle_weakness', 'puffy_face_and_eyes', 'mild_fever', 'skin_rash']))
