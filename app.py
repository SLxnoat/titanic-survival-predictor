import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c5f8a;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Load data and model
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/Titanic-Dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'Titanic-Dataset.csv' is in the data/ folder.")
        return None

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, feature_names
    except FileNotFoundError:
        st.error("Model file not found. Please run the training notebook first.")
        return None, None

# Preprocessing for single input
def preprocess_single_input(pclass, sex, age, sibsp, parch, fare, embarked):
    features = {}
    features['Pclass'] = pclass
    features['Sex'] = 1 if sex == 'male' else 0  # male=1, female=0
    features['Age'] = age
    features['SibSp'] = sibsp
    features['Parch'] = parch
    features['Fare'] = fare
    
    # Embarked encoding
    embarked_map = {'C': 0, 'Q': 1, 'S': 2}
    features['Embarked'] = embarked_map[embarked]
    
    # Family size & IsAlone
    features['FamilySize'] = sibsp + parch + 1
    features['IsAlone'] = 1 if features['FamilySize'] == 1 else 0
    
    # Title encoding
    if sex == 'male':
        if age < 16:
            features['Title'] = 2  # Master
        else:
            features['Title'] = 1  # Mr
    else:
        if age < 16:
            features['Title'] = 0  # Miss
        else:
            features['Title'] = 3  # Mrs
    
    return list(features.values())

# Main app
def main():
    train_df = load_data()
    model, feature_names = load_model()
    
    if train_df is None or model is None:
        st.stop()
    
    st.markdown('<h1 class="main-header">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict passenger survival using machine learning")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section:", 
                               ["🏠 Home", "📊 Data Exploration", "📈 Visualizations", 
                                "🔮 Make Prediction", "📉 Model Performance"])
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Exploration":
        show_data_exploration(train_df)
    elif page == "📈 Visualizations":
        show_visualizations(train_df)
    elif page == "🔮 Make Prediction":
        show_prediction(model, feature_names)
    elif page == "📉 Model Performance":
        show_model_performance(model, feature_names)

# Sections
def show_home():
    st.markdown("""
    ## Welcome to the Titanic Survival Predictor! 🚢
    This app predicts Titanic passenger survival chances based on personal details.
    
    **Features:**
    - 📊 Data Exploration
    - 📈 Interactive Visualizations
    - 🔮 Custom Predictions
    - 📉 Model Performance
    """)

def show_data_exploration(df):
    st.markdown('<h2 class="section-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Passengers", df.shape[0])
    with col2:
        st.metric("Survivors", df['Survived'].sum())
    with col3:
        st.metric("Survival Rate", f"{df['Survived'].mean():.1%}")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    st.subheader("Missing Values")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Column', 'Missing Count']
    st.dataframe(missing_df[missing_df['Missing Count'] > 0])

def show_visualizations(df):
    st.markdown('<h2 class="section-header">📈 Visualizations</h2>', unsafe_allow_html=True)
    fig1 = px.pie(df, names='Survived', title='Overall Survival Rate')
    st.plotly_chart(fig1, use_container_width=True)

def show_prediction(model, feature_names):
    st.markdown('<h2 class="section-header">🔮 Make a Prediction</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Gender", ["male", "female"])
        age = st.slider("Age", 0, 80, 30)
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
    with col2:
        parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
        fare = st.number_input("Fare", 0.0, 512.0, 32.0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
    
    if st.button("Predict Survival"):
        input_features = preprocess_single_input(pclass, sex, age, sibsp, parch, fare, embarked)
        pred = model.predict([input_features])[0]
        prob = model.predict_proba([input_features])[0]
        
        st.write("Prediction:", "Survived ✅" if pred == 1 else "Did Not Survive ❌")
        st.write("Survival Probability:", f"{prob[1]:.1%}")
        st.write("Death Probability:", f"{prob[0]:.1%}")

def show_model_performance(model, feature_names):
    st.markdown('<h2 class="section-header">📉 Model Performance</h2>', unsafe_allow_html=True)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

# Entry Point
if __name__ == "__main__":
    main()
