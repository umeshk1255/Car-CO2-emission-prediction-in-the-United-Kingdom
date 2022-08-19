from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModelshort1.pkl','rb'))
car=pd.read_csv('cleaned_car3.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['Manufacturer'].unique())
    car_models=sorted(car['Model'].unique())
    Engine_Capacity=sorted(car['Engine_Capacity'].unique())
    fuel_type=car['Fuel_Type'].unique()

    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, Engine_Capacity=Engine_Capacity,fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    Engine_Capacity=request.form.get('Engine_Capacity')
    fuel_type=request.form.get('fuel_type')
    cost=request.form.get('Annual_fuel_Cost_10000_Miles')

    prediction=model.predict(pd.DataFrame([[company,car_model, Engine_Capacity, fuel_type, cost]], columns=['Manufacturer', 'Model', 'Engine_Capacity',
     'Fuel_Type','Annual_fuel_Cost_10000_Miles']))
    
    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run()