from flask import Flask,request,render_template
import json 
import numpy as np
import pandas as pd
# from src.pipeline import predict

with open('./artifact/Unique_vals.json','r') as file:
    data=json.load(file)
col_fields = dict(zip(data.keys(), [value.split(',') for value in data.values()]))
with open('./artifact/num_col.json','r') as file:
    lis=json.load(file)
num_fields=list(lis)

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    else:
        print("Form POST initiated")
        dk=dict()
        for f in list(col_fields.keys()):
            dk[f]=list(request.form.get(f))
        for f in list(num_fields):
            dk[f]=list(request.form.get(f))
        print("Form data recieved")
        data=pd.DataFrame(dk)
        print("Form data converted to datframe")
        print('Predicting data')
        




        
