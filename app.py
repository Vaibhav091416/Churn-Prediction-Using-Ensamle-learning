from flask import Flask,request,render_template
import json 
import numpy as np
import pandas as pd
from src.pipeline.predict import PredictPipeline

with open('./artifact/Unique_vals.json','r') as file:
    data=json.load(file)
col_fields = dict(zip(data.keys(), [value.split(',') for value in data.values()]))
with open('./artifact/num_col.json','r') as file:
    lis=json.load(file)
num_fields=list(lis)

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html',col_fields=col_fields,num_fields=num_fields)
@app.route('/predictdata',methods=['GET','POST'])
def predict():
    print("entering the predict funciton")
    if request.method=='GET':
        return render_template('index.html')
    else:
        print("Form POST initiated")
        try:
            dk=dict()
            for f in list(col_fields.keys()):
                dk[f]=request.form.get(f)
            for f in list(num_fields):
                dk[f]=request.form.get(f)
            print("Form data recieved")
            data=pd.DataFrame([dk])
            print(data)
            print(type(data))
            print("Form data converted to datframe")
            print('Predicting data')
            obj=PredictPipeline()
            transformed_data=obj.data_transform(data)
            ans=obj.predict(transformed_data)
            print("predicted data",ans)
            ans_st="-1"
            if ans==0:
                ans_st='No'
            else:
                ans_st="Yes"
            return render_template('index.html',col_fields=col_fields,num_fields=num_fields,ans_st=ans_st)
        except Exception as e:
            print("Error: ",e)
            return render_template('index.html',col_fields=col_fields,num_fields=num_fields,ans_st="error")

if __name__=='__main__':
    app.run(debug=True)
        




        
