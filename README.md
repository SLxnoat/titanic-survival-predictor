# Titanic Survival Predictor 🚢

A complete machine learning pipeline from data exploration to deployment, predicting Titanic passenger survival with 82% accuracy.

## Features

📊 Data Exploration
- View raw passenger data
- Summary statistics
- Missing value analysis

📈 Interactive Visualizations
- Survival rate pie charts
- Class/gender survival comparisons
- Feature importance analysis

🔮 Predictive Modeling
- Random Forest classifier (82% accuracy)
- Logistic Regression baseline
- Probability estimates

🚀 Web Application
- Streamlit interactive interface
- Mobile-responsive design
- Real-time predictions

## Installation

1. Clone the repository:
   git clone [https://github.com/yourusername/titanic-survival-predictor.git](https://github.com/SLxnoat/titanic-survival-predictor)
   cd titanic-survival-predictor

2. Set up virtual environment:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the application:
   streamlit run app.py

## Project Structure

titanic-survival-predictor/
├── app.py                  # Main Streamlit application
├── data/
│   └── Titanic-Dataset.csv  # Titanic passenger dataset
├── feature_names.pkl       # Saved feature names
├── model.pkl               # Trained Random Forest model
├── notebooks/
│   └── model_training.ipynb # Complete training notebook
├── README.md               # Documentation
└── requirements.txt        # Python dependencies

## Data Processing

Key Preprocessing Steps:
1. Handle missing values (Age, Embarked, Fare)
2. Feature engineering:
   - FamilySize = SibSp + Parch + 1
   - IsAlone flag
   - Title extraction from names
3. Categorical encoding (Sex, Embarked, Title)

## Model Training

Algorithms Compared:
- Random Forest (100 trees) - 82.1% accuracy
- Logistic Regression - 79.3% accuracy

Top Predictive Features:
1. Sex (44% importance)
2. Fare (18%)
3. Age (14%)
4. Passenger Class (9%)

## Web Application

Navigation Sections:
1. Home - Project overview
2. Data Exploration - Raw data analysis
3. Visualizations - Interactive charts
4. Make Prediction - Survival calculator
5. Model Performance - Feature importance

Prediction Inputs:
- Passenger class (1-3)
- Gender
- Age (slider)
- Family members
- Fare amount
- Embarkation port

## Usage Example

1. Select passenger details in sidebar
2. Click "Predict Survival"
3. View results:
   - Survival prediction (✅/❌)
   - Survival probability percentage
   - Death probability percentage

## API Extension Example

The app can be extended with Flask API endpoints:

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    processed = preprocess_input(data)
    prediction = model.predict([processed])[0]
    return jsonify({
        'survived': bool(prediction),
        'probability': model.predict_proba([processed])[0][1]
    })

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

Suggested Improvements:
- Add more visualization types
- Include passenger cabin analysis
- Deploy as Docker container
- Add authentication layer

## License


MIT License
