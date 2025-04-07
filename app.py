import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page title and configuration
st.set_page_config(
    page_title="Medical Diagnosis System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Create directory for static files if it doesn't exist
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Load CSS from external file
@st.cache_data
def load_css(file_name):
    try:
        with open(os.path.join(static_dir, file_name)) as f:
            return f.read()
    except FileNotFoundError:
        return """
        /* Default styles if CSS file is not found */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #0D47A1;
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1976D2;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        /* More default styles can be added here */
        """

# Apply the CSS
st.markdown(f'<style>{load_css("styles.css")}</style>', unsafe_allow_html=True)

# Page header with logo
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<div class="main-header">🏥 Medical Diagnosis and Recommendation System</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #555; margin-bottom: 30px;">
        Enter your symptoms to get disease prediction and medicine recommendations
    </p>
    """, unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    models = {}
    model_dir = "models"
    
    if not os.path.exists(model_dir):
        st.error("Model directory not found. Please train and save models first.")
        return None
    
    try:
        # Load all available models
        for model_file in os.listdir(model_dir):
            if model_file.endswith('.pkl') and model_file != 'symptoms.pkl':  # Skip the symptoms file
                model_name = model_file.split('.')[0]
                with open(os.path.join(model_dir, model_file), 'rb') as f:
                    models[model_name] = pickle.load(f)
        
        if not models:
            st.error("No models found in the models directory.")
            return None
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load symptom list from the saved file with caching
@st.cache_data
def load_symptoms():
    try:
        with open('models/symptoms.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading symptoms: {e}")
        # Fallback to a default list if the file can't be loaded
        return [
            "fever", "cough", "fatigue", "shortness of breath", "headache",
            "sore throat", "body aches", "runny nose", "nausea", "vomiting",
            "diarrhea", "chest pain", "abdominal pain", "rash", "joint pain",
            "dizziness", "chills", "loss of appetite", "weight loss", "swelling"
        ]

# Load medicine database with caching
@st.cache_data
def load_medicine_database():
    # In a real application, this would load from a database or file
    # For now, we'll use a sample dictionary
    return {
        "Common Cold": ["Acetaminophen", "Ibuprofen", "Decongestant"],
        "Influenza": ["Oseltamivir", "Zanamivir", "Acetaminophen"],
        "COVID-19": ["Acetaminophen", "Ibuprofen", "Rest and fluids"],
        "Pneumonia": ["Antibiotics", "Cough medicine", "Pain relievers"],
        "Bronchitis": ["Antibiotics", "Cough suppressants", "Bronchodilators"],
        "Sinusitis": ["Antibiotics", "Decongestants", "Pain relievers"],
        "Gastroenteritis": ["Oral rehydration", "Probiotics", "Anti-diarrheals"],
        "Migraine": ["Triptans", "NSAIDs", "Anti-nausea medication"],
        "Hypertension": ["ACE inhibitors", "Beta blockers", "Diuretics"],
        "Diabetes": ["Insulin", "Metformin", "Sulfonylureas"]
    }

# Function to predict disease based on symptoms
def predict_disease(symptoms, models):
    # Prepare input data
    all_symptoms = load_symptoms()
    input_data = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    
    # Make predictions with all models
    predictions = {}
    for model_name, model in models.items():
        try:
            prediction = model.predict([input_data])[0]
            probability = max(model.predict_proba([input_data])[0]) * 100
            predictions[model_name] = (prediction, probability)
        except Exception as e:
            st.error(f"Error with model {model_name}: {e}")
    
    return predictions

# Function to recommend medicine based on predicted disease
def recommend_medicine(disease):
    medicine_db = load_medicine_database()
    return medicine_db.get(disease, ["No specific medication found. Please consult a doctor."])

# Function to create a pie chart of model predictions
@st.cache_data
def create_prediction_chart(predictions):
    # Extract disease names and create a frequency count
    diseases = [pred[0] for _, pred in predictions.items()]
    disease_counts = {}
    for disease in diseases:
        if disease in disease_counts:
            disease_counts[disease] += 1
        else:
            disease_counts[disease] = 1
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        disease_counts.values(), 
        labels=disease_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        colors=sns.color_palette('pastel')
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    plt.title('Disease Prediction Distribution', fontsize=16, pad=20)
    
    # Style the labels and percentages
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    return fig

# Main application
def main():
    # Create sidebar for additional information
    with st.sidebar:
        st.image("https://img.freepik.com/free-vector/medical-healthcare-blue-background_1017-26807.jpg", use_column_width=True)
        st.markdown("## About")
        st.info("""
        This application uses machine learning to predict possible diseases based on symptoms and provides medicine recommendations.
        
        **How to use:**
        1. Select at least 3 symptoms
        2. Click the Diagnose button
        3. Review the predictions and recommendations
        
        **Note:** This is for educational purposes only.
        """)
        
        st.markdown("## Models Used")
        st.markdown("""
        - Random Forest
        - Naive Bayes
        - Support Vector Machine (SVM)
        - K-Nearest Neighbors (KNN)
        - Neural Network
        """)
    
    # Load models
    models = load_models()
    
    # If models failed to load, show a message and exit
    if models is None:
        st.warning("⚠️ Please train and save models before using this application.")
        st.markdown("""
        ### How to train models:
        Run the following command in your terminal:
        ```
        python train_models.py
        ```
        """)
        return
    
    # Get available symptoms
    available_symptoms = load_symptoms()
    
    # Create symptom input section
    st.markdown('<div class="sub-header">Enter Your Symptoms</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Create 5 symptom selection dropdowns
    selected_symptoms = []
    cols = st.columns(5)
    
    for i in range(5):
        with cols[i]:
            st.markdown(f'<p style="font-weight: bold; color: #1976d2;">Symptom {i+1}</p>', unsafe_allow_html=True)
            symptom = st.selectbox(
                "",
                options=[""] + available_symptoms,
                key=f"symptom_{i}",
                label_visibility="collapsed"
            )
            if symptom:
                selected_symptoms.append(symptom)
    
    # Display selected symptoms summary
    if selected_symptoms:
        st.markdown('<div class="selected-symptoms-container">', unsafe_allow_html=True)
        st.markdown('<div class="selected-symptoms-header">Selected Symptoms</div>', unsafe_allow_html=True)
        # Create a container for the symptoms
        symptoms_html = ""
        for symptom in selected_symptoms:
            symptoms_html += f'<span class="selected-symptom">{symptom.title()}</span> '
        
        st.markdown(f'<div>{symptoms_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a diagnose button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        diagnose_button = st.button("Diagnose", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if diagnose_button:
        if len(selected_symptoms) < 3:
            st.warning("⚠️ Please enter at least 3 symptoms for accurate diagnosis.")
        else:
            # Make prediction
            with st.spinner("Analyzing symptoms..."):
                predictions = predict_disease(selected_symptoms, models)
            
            if predictions:
                # Display results
                st.markdown('<div class="sub-header">Diagnosis Results</div>', unsafe_allow_html=True)
                
                # Create columns for each model's prediction
                result_cols = st.columns(len(predictions))
                
                for i, (model_name, (disease, probability)) in enumerate(predictions.items()):
                    with result_cols[i]:
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="model-name">{model_name}</div>
                            <div class="disease-name">{disease}</div>
                            <div class="confidence">Confidence: {probability:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Create and display the prediction chart
                st.markdown("### Prediction Distribution")
                fig = create_prediction_chart(predictions)
                st.pyplot(fig)
                
                # Get the most common prediction or the one with highest confidence
                # Find the most frequent disease prediction
                disease_counts = {}
                disease_confidences = {}
                
                for _, (disease, confidence) in predictions.items():
                    if disease in disease_counts:
                        disease_counts[disease] += 1
                        disease_confidences[disease] += confidence
                    else:
                        disease_counts[disease] = 1
                        disease_confidences[disease] = confidence
                
                # Find the disease with the highest count, or highest average confidence if tied
                max_count = max(disease_counts.values())
                most_likely_diseases = [d for d, c in disease_counts.items() if c == max_count]
                
                if len(most_likely_diseases) > 1:
                    # If there's a tie, use the one with highest average confidence
                    predicted_disease = max(
                        most_likely_diseases, 
                        key=lambda d: disease_confidences[d] / disease_counts[d]
                    )
                else:
                    predicted_disease = most_likely_diseases[0]
                
                # Recommend medicine
                st.markdown('<div class="sub-header">Medicine Recommendations</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    Based on the analysis, the most likely condition is <b style="color: #0D47A1; font-size: 1.2rem;">{predicted_disease}</b>
                </div>
                """, unsafe_allow_html=True)
                
                medicines = recommend_medicine(predicted_disease)
                
                for medicine in medicines:
                    st.markdown(f"""
                    <div class="medicine-item">
                        <span style="font-weight: bold;">💊</span> {medicine}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="disclaimer">
                    <h4 style="color: #d32f2f; margin-top: 0;">⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
                    <p>This application provides general information only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
                    <p>Never disregard professional medical advice or delay in seeking it because of something you have read on this application.</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
