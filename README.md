# SGPA and CGPA Prediction Project

## Overview
This project focuses on predicting the SGPA (Semester Grade Point Average) and CGPA (Cumulative Grade Point Average) of students in their 5th semester. It involves data preprocessing, exploratory data analysis (EDA), machine learning model training, and deployment of a web application for interactive prediction.

## Repository Contents

- `app.py`: A Streamlit-based web application for predicting SGPA and CGPA.
- `data_and_model.ipynb`: A Jupyter notebook containing data preprocessing, EDA, model training, and evaluation.
- `FinalizedDataset.xlsx`: The dataset used for training and testing the models.
- `label_encoders.pkl`, `min_max_scaler.pkl`: Preprocessing objects saved for use in the application.
- `preprocessed_data.xlsx`: The dataset after preprocessing.
- `rf_sgpa5_model.pkl`, `ridge_cgpa5_model.pkl`: Trained machine learning models for SGPA and CGPA prediction.

## Data Preprocessing
The data preprocessing steps include handling missing values, normalizing numerical data, and encoding categorical data. This ensures that the dataset is clean and ready for model training.

## Exploratory Data Analysis (EDA)
EDA is performed to understand the distributions, correlations, and other insights from the data, which is crucial for model building.

## Machine Learning Models
Several models are trained and evaluated, including Linear Regression, Random Forest, Gradient Boosting, Support Vector Regression, and Neural Networks. The best-performing models for SGPA and CGPA prediction are saved for deployment.

## Web Application
The Streamlit web application (`app.py`) provides a user-friendly interface for inputting student data and obtaining SGPA and CGPA predictions.

## Usage
To run the web application locally:
1. Ensure Python and necessary packages (Streamlit, Pandas, Numpy, Scikit-learn, etc.) are installed.
2. Run `streamlit run app.py` in your terminal.
3. The application will open in your web browser.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.
