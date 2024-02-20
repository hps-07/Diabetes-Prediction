from flask import Flask,render_template,request,redirect
app=Flask(__name__,template_folder="templates")
import pickle
import numpy as np

model = pickle.load(open("diabetes-prediction-logreg-model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/About")
def about():
    return render_template("about.html")

@app.route("/Research")
def research():
    return render_template("research.html")

@app.route("/ContactUs")
def contact():
    return render_template("contact.html")

@app.route("/Login")
def login():
    return render_template("login.html")

@app.route("/Register")
def register():
    return render_template("registration.html")

@app.route("/Diagnosis",methods = ['GET', 'POST'])
def diagnosis():
    return render_template("diagnosis.html")    


# Diagnosis/Prediction part starts

@app.route("/predict", methods=['POST'])
def predict():
    Pregnancies = request.form['Pregnancies']
    Glucose = request.form['Glucose']
    BloodPressure = request.form['BloodPressure']
    BMI = request.form['BMI']
    DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    Age = request.form['Age']

    Pregnancies= int(Pregnancies)
    Glucose= int(Glucose)
    BloodPressure = int(BloodPressure)
    BMI = float(BMI)
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = int(Age)

    final_features = np.array([(Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age)])
    prediction = model.predict(final_features)
    return render_template('diagnosis.html', prediction_text = "The patient has diabetes : {}".format(prediction))



    #return render_template("diagnosis.html")


# Diagnosis/Prediction part Ends

if __name__ == '__main__':
    app.run(debug=True,port=2222)