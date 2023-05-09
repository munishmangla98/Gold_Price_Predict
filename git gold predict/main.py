from flask import Flask, render_template,request,url_for
import pickle
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def gold():
   return render_template('gold.html')

@app.route('/predict', methods=['POST'])
def predict():
    spx=request.form['proname']
    uso=request.form['proct']
    svl=request.form['proid']
    usd=request.form['proid1']
    result=round(model.predict([[spx,uso,svl,usd]])[0],2)
    return render_template('gold.html',result=result)




app.run(debug=True)
