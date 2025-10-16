# Heart-Attack-Predictor

Technical Summary – 
1. Reviews
Cardiovascular diseases remain one of the leading causes of death worldwide. Early prediction of heart attack risk can save lives through preventive measures and timely medical attention. The proposed model leverages machine learning techniques to analyse health parameters and lifestyle factors, providing a risk assessment for potential heart attacks. It utilises a synthetic dataset with multiple health-related features to predict whether a person is at low, medium, or high risk of a heart attack.
________________________________________
2. Objective
The primary objective of this project is to develop a machine learning model that predicts the risk of a heart attack based on a person's health indicators, lifestyle habits, and medical history.
The model aims to:
•	Identify individuals at high risk.
•	Help doctors and individuals take preventive action.
•	Demonstrate how data-driven insights can assist in healthcare decision-making.
________________________________________
3. Dataset
The dataset used is “Heart Attack Prediction Dataset.xlsx”, containing health and demographic information for multiple individuals.
It includes features such as:
•	Numerical Attributes: Age, Cholesterol, Blood Pressure, Heart Rate, Stress Level, Triglycerides, etc.
•	Categorical Attributes: Sex, Country, Diet, Hemisphere, etc.
•	Binary/Indicator Features: Diabetes, Smoking, Family History, Obesity, Alcohol Consumption, Previous Heart Problems, etc.
•	Target Column: Heart Attack Risk (Low / Medium / High)
This combination of attributes provides a comprehensive view of the factors that influence the likelihood of a heart attack.
________________________________________
4. Technical Workflow

The workflow of the project consists of the following key steps:
1.	Data Loading and Preprocessing
•	Loaded the dataset using pandas.
•	Handled missing values and categorized data into numerical and categorical features.
2.	Feature Encoding and Scaling
•	Categorical data converted into numerical format using OneHotEncoder.
•	Numerical features standardized using StandardScaler.
3.	Data Splitting
•	Split data into training (80%) and testing (20%) sets using train_test_split.
4.	Model Selection and Training
•	Used a Random Forest Classifier (from sklearn.ensemble) to train the model.
•	The model learns complex patterns between health attributes and heart attack risk.
5.	Evaluation
•	Checked model performance using metrics such as accuracy and classification report.
•	Fine-tuned using different random states and parameter adjustments.
6.	User Input Prediction
•	Model takes new user input (Age, Sex, Cholesterol, etc.) and predicts the risk level.
•	Outputs both the predicted class (Low/Medium/High) and the probability score.
________________________________________
5. Tools and Libraries
Library	Purpose
pandas	Data manipulation and analysis
numpy	Numerical computations
scikit-learn (sklearn)	Machine learning algorithms and preprocessing
openpyxl	Reading Excel (.xlsx) files
LabelEncoder, OneHotEncoder, StandardScaler	Data encoding and normalisation
RandomForestClassifier	Prediction model for classification tasks
________________________________________
6. Key Learnings
•	Understood how health parameters can influence heart attack risk.
•	Learned how to preprocess mixed-type (numerical + categorical) data effectively.
•	Gained experience with encoding, scaling, and model evaluation techniques.
•	Explored the interpretability of Random Forests and feature importance.
•	Learned how to deploy a model that accepts real-time user input for prediction.
________________________________________
 
7. Conclusion
The developed model accurately predicts the risk level of heart attacks based on various health and lifestyle factors.
It demonstrates how machine learning can assist in preventive healthcare by identifying potential risk groups.
While the model performs well on the given dataset, future improvements could include:
•	Collecting real-world medical data for better accuracy.
•	Integrating the system into a web or mobile app for user-friendly access.
•	Using explainable AI techniques to make predictions more transparent.

Output screenshot - 
<img width="1056" height="393" alt="image" src="https://github.com/user-attachments/assets/428ee39d-3845-45d7-b5c7-d559252097b7" />
<img width="804" height="590" alt="image" src="https://github.com/user-attachments/assets/38dd2c0d-f414-4736-8060-3bb176cd9ede" />
<img width="639" height="721" alt="image" src="https://github.com/user-attachments/assets/dfe3cf45-3bdb-41ee-8efe-458f6b4754c0" />

