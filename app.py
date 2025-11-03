import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import os
model_path = os.path.join(os.path.dirname(__file__), 'placement_model.pkl')
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def index():
    return render_template('home.html')

@app.route('/results')
def results():
    return render_template('result.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()  # ✅ Read JSON payload
        cgpa = float(data['cgpa'])
        projects = int(data['projects_count'])
        internships = int(data['internships_count'])
        certs = int(data['cert_count'])
        extra = float(data['extracurricular_score'])
        soft = float(data['softskill_rating'])
        # name = request.form['name']
        # branch = request.form['branch']
        # gender = request.form['gender']
        # year = int(request.form['year'])
        # attendance = float(request.form['attendance_pct'])
        # aptitude = float(request.form['aptitude_score'])
        # technical = float(request.form['technical_score'])
        # comm = float(request.form['communication_score'])
        # mock = float(request.form['mock_interview_score'])
        
        # Prepare input
        features = np.array([[cgpa, projects, internships, certs, extra, soft]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100
        confidence = round(probability, 2)
        # Recommended track
        # if technical > 70 and aptitude > 65:
        #     track = "Software Developer"
        # elif comm > 70:
        #     track = "HR / Communication Specialist"
        # elif aptitude > 75:
        #     track = "Data Analyst"
        # else:
        #     track = "Technical Support / QA"

        # Skill improvement suggestions
        if confidence < 50:
            next_step = "Focus on improving your technical and communication skills."
        elif confidence < 80:
            next_step = "You’re doing great! Sharpen project experience and mock interviews."
        else:
            next_step = "Excellent! Keep practicing aptitude and technical tests."

        return jsonify({
            "status": "success",
            "prediction": "Placed" if prediction == 1 else "Not Placed",
            "confidence": confidence,
            "recommendations": [next_step]
        })

        return render_template(
            'result.html',
            result=prediction,
            probability=round(probability, 2),
            next_step=next_step,
            confidence=confidence
        )

if __name__ == "__main__":
    app.run(debug=True)
