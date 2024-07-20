from flask import Flask,render_template,request
import pickle
import numpy as np

model_path="model.pkl"

with open(model_path,'rb') as file:
    model=pickle.load(file)
    
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict",methods=['POST'])
def predict():

    #get input values
    features=[float(x) for x in request.form.values()]
    final_featres=[np.array(features)]

    #make prediction
    prediction =model.predict(final_featres)
    package = round(prediction[0],2)
    output = f"Predicted Package is {package} lakhs"

    return render_template("predict.html",msg=output)


if __name__=="__main__":
    app.run(debug=True)