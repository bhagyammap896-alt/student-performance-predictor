import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')

# Load the data to get feature names and sample
df = pd.read_csv('Student_Performance.csv')
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Example prediction: use the first row as sample input
sample = df.drop('Performance Index', axis=1).iloc[0:1]
actual = df['Performance Index'].iloc[0]

prediction = model.predict(sample)
print(f'Sample Input: {sample.values.tolist()}')
print(f'Actual Performance Index: {actual}')
print(f'Predicted Performance Index: {prediction[0]:.2f}')

# You can input your own values here
# For example: Hours Studied=6, Previous Scores=85, Extracurricular=1, Sleep Hours=8, Sample Papers=3
custom_input = pd.DataFrame([[6, 85, 1, 8, 3]], columns=['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced'])
custom_prediction = model.predict(custom_input)
print(f'Custom Input Prediction: {custom_prediction[0]:.2f}')