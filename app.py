from flask import Flask, render_template, jsonify,redirect,request
from flask_pymongo import PyMongo
import pandas as pd
import numpy as np
import _pickle as cPickle
import os


# MONGODB_URI = os.environ.get('MONGODB_URI')
# if not MONGODB_URI:
#     MONGODB_URI = "mongodb://localhost:5000/"

app = Flask(__name__)


# app.config['MONGO_URI'] = MONGODB_URI

mongo = PyMongo(app)


@app.route("/")
def index ():

   
    return render_template('index.html', location =  "Back Cover")


@app.route("/send", methods=["GET", "POST"])
def predictions():

    mongo.db.resultsDB.remove()

    if request.method == "POST":
        creditPolicy = request.form['creditPolicy']
        purpose =  request.form['purpose']
        interestRate  =  request.form['interestRate']
        installment =  request.form['installment']
        logAnnual =  request.form['logAnnual']
        dti =  request.form['dti']
        fico =  request.form['fico']
        creditLine =  request.form['creditLine']
        revolvingBalance =  request.form['revolvingBalance']
        revolvingUtilizationRate =  request.form['revolvingUtilizationRate']
        inqLast6mths =  request.form['inqLast6mths']
        delinq2yrs =  request.form['delinq2yrs']
        pubRec =  request.form['pubRec']


    df_loan = pd.read_csv('loan_data.csv')

    with open('RFMODEL','rb') as Model:
        rfK = cPickle.load(Model)

    df_loan = df_loan.append({'credit.policy': np.float32(creditPolicy), 
                'purpose': purpose,
                'int.rate': np.float32(interestRate) , 
                'installment': np.float32(installment), 
                'log.annual.inc': np.float32(logAnnual),
                'dti': np.float32(dti), 
                'fico': np.float32(fico),
                'days.with.cr.line':np.float32(creditLine),
                'revol.bal': np.float32(revolvingBalance), 
                'revol.util': np.float32(revolvingUtilizationRate),
                'inq.last.6mths': np.float32(inqLast6mths), 
                'delinq.2yrs': np.float32(delinq2yrs), 
                'pub.rec': np.float32(pubRec), 
                'not.fully.paid': 1 # constant value
                            },ignore_index=True)

    results = {}

    results['vector'] = pd.get_dummies(df_loan.drop('not.fully.paid',axis=1)).iloc[-1:].to_dict('records')

    x = pd.get_dummies(df_loan.drop('not.fully.paid',axis=1))

    xVector = x.iloc[-1:]

    results['prediction'] = rfK.predict(xVector).item()

    mongo.db.resultsDB.insert({"vectorsANDPREDICT": results}, check_keys = False)


    return redirect('/', code=302) 



@app.route("/data/results")
def targetVALUES():

    targetANDPREDICTLIST = []

    for userresults in mongo.db.resultsDB.find():
        userresults.pop("_id")

        targetANDPREDICTLIST.append(userresults)

    
    
    
    
    
    return  jsonify(targetANDPREDICTLIST)


if __name__ == "__main__":
    app.run(debug=True)
