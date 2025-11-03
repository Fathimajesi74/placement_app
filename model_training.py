import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("students_synth.csv")

# Encode categorical columns
label_encoders = {}
for col in ['branch', 'gender', 'placed_role']:
    le = LabelEncoder()
    df[col] = df[col].fillna("Unknown")
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = df.drop(columns=['student_id', 'is_placed'])  # drop non-feature + target
y = df['is_placed']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("✅ Model Training Completed!")
print("🔹 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("placement_model.pkl", "wb"))

# Save label encoders for later use
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

print("\n🎉 Model and encoders saved successfully!")
