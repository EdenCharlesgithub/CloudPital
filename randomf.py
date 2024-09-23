from sklearn.ensemble import RandomForestClassifier  # Importing RandomForest for ensemble learning
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import pandas as pd
import os

# Load and prepare the dataset
data = pd.read_csv(os.path.join("templates", "Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print("RandomForest - Ensemble Learning")
# Use RandomForestClassifier instead of DecisionTreeClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf = rf.fit(x_train, y_train)

# # Uncomment to check the accuracy
print("Accuracy: ", clf_rf.score(x_test, y_test))

# Assign indices to symptoms
indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]
dictionary = dict(zip(symptoms, indices))

# Function to predict based on user input symptoms
def dosomething(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1, 1)).transpose()
    return rf.predict(user_input_label)

# Example usage of dosomething function with ensemble learning
# print(dosomething(['headache', 'muscle_weakness', 'puffy_face_and_eyes', 'mild_fever', 'skin_rash']))
# prediction = []
# for i in range(7):
#     pred = dosomething(['headache'])
#     prediction.append(pred)
# print(prediction)
