from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# HTML template
template = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Performance Predictor</title>
</head>
<body>
    <h1>Student Performance Index Predictor</h1>
    <form method="POST" action="/predict">
        <label>Hours Studied:</label>
        <input type="number" name="hours_studied" required><br><br>
        
        <label>Previous Scores:</label>
        <input type="number" name="previous_scores" required><br><br>
        
        <label>Extracurricular Activities:</label>
        <select name="extracurricular">
            <option value="1">Yes</option>
            <option value="0">No</option>
        </select><br><br>
        
        <label>Sleep Hours:</label>
        <input type="number" name="sleep_hours" required><br><br>
        
        <label>Sample Question Papers Practiced:</label>
        <input type="number" name="sample_papers" required><br><br>
        
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
    <h2>Predicted Performance Index: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(template)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    hours_studied = int(request.form['hours_studied'])
    previous_scores = int(request.form['previous_scores'])
    extracurricular = int(request.form['extracurricular'])
    sleep_hours = int(request.form['sleep_hours'])
    sample_papers = int(request.form['sample_papers'])
    
    # Create DataFrame for prediction
    input_data = pd.DataFrame([[hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers]], 
                              columns=['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced'])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    return render_template_string(template, prediction=f"{prediction:.2f}")

if __name__ == '__main__':
    app.run(debug=True)