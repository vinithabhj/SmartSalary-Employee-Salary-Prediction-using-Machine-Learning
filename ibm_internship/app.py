from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model pipeline (make sure this .pkl file is in the same folder)
model = pickle.load(open('smartsalary_employee_salary_prediction_using_machine_learning.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    experience_level = request.form['experience_level']
    employment_type = request.form['employment_type']
    job_title = request.form['job_title']
    remote_ratio = float(request.form['remote_ratio'])
    company_size = request.form['company_size']

    # Prepare input DataFrame for prediction
    input_df = pd.DataFrame({
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'remote_ratio': [remote_ratio],
        'company_size': [company_size]
    })

    # Predict salary
    predicted_salary = model.predict(input_df)[0]

    # Return result page with prediction
    return render_template('result.html', salary=round(predicted_salary, 2))

if __name__ == '__main__':
    app.run(debug=True)
