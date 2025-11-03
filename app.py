import numpy as np
from flask import Flask, render_template, request
import pickle
import os
model_path = os.path.join(os.path.dirname(__file__), 'placement_model.pkl')
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        branch = request.form['branch']
        gender = request.form['gender']
        year = int(request.form['year'])
        cgpa = float(request.form['cgpa'])
        attendance = float(request.form['attendance_pct'])
        aptitude = float(request.form['aptitude_score'])
        technical = float(request.form['technical_score'])
        comm = float(request.form['communication_score'])
        projects = int(request.form['projects_count'])
        internships = int(request.form['internships_count'])
        certs = int(request.form['cert_count'])
        extra = float(request.form['extracurricular_score'])
        mock = float(request.form['mock_interview_score'])
        soft = float(request.form['softskill_rating'])

        # Prepare input
        features = np.array([[year, cgpa, attendance, aptitude, technical, comm,
                              projects, internships, certs, extra, mock, soft]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(
            features)[0][1] * 100  # Placement probability
        confidence = round(probability, 2)

        # Recommended track
        if technical > 70 and aptitude > 65:
            track = "Software Developer"
        elif comm > 70:
            track = "HR / Communication Specialist"
        elif aptitude > 75:
            track = "Data Analyst"
        else:
            track = "Technical Support / QA"

        # Skill improvement suggestions
        if confidence < 50:
            next_step = "Focus on improving your technical and communication skills."
        elif confidence < 80:
            next_step = "You’re doing great! Sharpen project experience and mock interviews."
        else:
            next_step = "Excellent! Keep practicing aptitude and technical tests."

        return render_template(
            'result.html',
            name=name,
            result=prediction,
            probability=round(probability, 2),
            track=track,
            next_step=next_step,
            confidence=confidence
        )

if __name__ == "__main__":
    app.run(debug=True)
