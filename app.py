from flask import Flask,render_template,request,send_from_directory
import numpy as np
import joblib

# Load Pickle Files
scaler = joblib.load("./models/scaler.pkl")
rf_model = joblib.load("./models/rf_model.pkl")



app=Flask(__name__)

# -------------- HOME --------------------
@app.route('/', methods=['GET','POST'])
def home():
    return render_template('home.html')

# -------------- ABOUT --------------------
@app.route('/about', methods=['GET','POST'])
def about():
    return render_template('about.html')

# -------------- Heart Failure Prediction API --------------------
@app.route('/predicthfp', methods=['GET','POST'])
def predictcct():
    return render_template('predicthfp.html')

@app.route("/predictresulthfp", methods = ["GET", "POST"])
def predictcctresult():
    float_features = [np.float64(x) for x in request.form.values()]
    final = np.array([float_features])
    final_scaled = scaler.transform(final)
    prediction = rf_model.predict(final_scaled)
    print(prediction)
    return render_template('result.html', pred=prediction)

# -------------- Full Code Available --------------------
@app.route('/fullcode', methods=['GET','POST'])
def fullcode():
    return render_template('fullcode.html')

#--------------- Download Dataset -----------------------
@app.route('/downloaddataset')
def download_dataset():
    return send_from_directory(
        'dataset',
        'heart_failure_clinical_records_dataset.csv',
        as_attachment=True
    )

if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)
