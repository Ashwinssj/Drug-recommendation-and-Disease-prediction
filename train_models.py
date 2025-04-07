import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Create a directory to save models
if not os.path.exists('models'):
    os.makedirs('models')

# Sample data generation (in a real scenario, you would load your dataset)
def generate_sample_data():
    # List of symptoms
    symptoms = [
        "fever", "cough", "fatigue", "shortness of breath", "headache",
        "sore throat", "body aches", "runny nose", "nausea", "vomiting",
        "diarrhea", "chest pain", "abdominal pain", "rash", "joint pain",
        "dizziness", "chills", "loss of appetite", "weight loss", "swelling"
    ]
    
    # List of diseases
    diseases = [
        "Common Cold", "Influenza", "COVID-19", "Pneumonia", "Bronchitis",
        "Sinusitis", "Gastroenteritis", "Migraine", "Hypertension", "Diabetes"
    ]
    
    # Generate sample data
    num_samples = 1000
    data = []
    
    for _ in range(num_samples):
        # Randomly select a disease
        disease = np.random.choice(diseases)
        
        # Generate symptoms based on the disease (simplified approach)
        symptom_values = np.zeros(len(symptoms))
        
        # Assign some symptoms based on the disease (simplified)
        if disease == "Common Cold":
            symptom_values[symptoms.index("fever")] = np.random.choice([0, 1], p=[0.5, 0.5])
            symptom_values[symptoms.index("cough")] = np.random.choice([0, 1], p=[0.2, 0.8])
            symptom_values[symptoms.index("runny nose")] = np.random.choice([0, 1], p=[0.1, 0.9])
            symptom_values[symptoms.index("sore throat")] = np.random.choice([0, 1], p=[0.3, 0.7])
        elif disease == "Influenza":
            symptom_values[symptoms.index("fever")] = np.random.choice([0, 1], p=[0.1, 0.9])
            symptom_values[symptoms.index("cough")] = np.random.choice([0, 1], p=[0.2, 0.8])
            symptom_values[symptoms.index("body aches")] = np.random.choice([0, 1], p=[0.2, 0.8])
            symptom_values[symptoms.index("fatigue")] = np.random.choice([0, 1], p=[0.1, 0.9])
        # Add more disease-symptom relationships for other diseases
        
        # Add some random symptoms
        random_symptoms = np.random.choice(range(len(symptoms)), size=np.random.randint(0, 3), replace=False)
        for idx in random_symptoms:
            symptom_values[idx] = 1
        
        # Add to data
        row = list(symptom_values) + [disease]
        data.append(row)
    
    # Create DataFrame
    columns = symptoms + ["disease"]
    df = pd.DataFrame(data, columns=columns)
    
    return df

# Load or generate data
print("Generating sample data...")
df = generate_sample_data()

# Prepare features and target
X = df.drop('disease', axis=1)
y = df['disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to train
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'NaiveBayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# Train and save models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Save model
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"{name} model saved.")

print("All models trained and saved successfully!")

# Save the symptom list for reference
with open('models/symptoms.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("Symptom list saved.")