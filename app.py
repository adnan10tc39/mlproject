from flask import Flask,request,render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler #for preprocessor.pkl file
from src.pipeline.predict_pipeline import Customdata, PredictPipeline

application=Flask(__name__) #this Flask(__name__) give us a entry point


app = application

#route for our home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])#ye do method ko support karta hai get and post 
def predict_datapoint():
    #in this function we will get the data and perform prediction
    if request.method=='GET':
        return render_template('home.html') #it will render to our default home page.is home.html mien just data field majood houn gi,button etc
    else: #if it is not get then it will be post
        #here in post part we will capture the data,standard scaling etc.
        data=Customdata(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
            





