import pickle as pkl
import pandas as pd
import os 
import json
#all the categorical columns
with open('./artifact/Unique_vals.json','r') as file:
    data=json.load(file)
col_fields = dict(zip(data.keys(), [value.split(',') for value in data.values()]))

#All the numerical columns
with open('./artifact/num_col.json','r') as file:
    lis=json.load(file)
num_fields=list(lis)

#Loading the model
with open('./artifact/final_model_abcl.pkl',"rb") as file_obj:
    model=pkl.load(file_obj)

#Loading the label_encoder
with open('./artifact/label_encoder.pkl','rb') as file_obj:
    le=pkl.load(file_obj)

#Loading the standard sclaer
with open('./artifact/scaler.pkl','rb') as file_obj:
    le=pkl.load(file_obj)

cat_cols=list(col_fields.keys())
bin_col=[col for col in list(col_fields.keys()) if len(col_fields[col])==2]
multi_col=[col for col in list(col_fields.keys()) if len(col_fields[col])>2]
model_feauture_list=list(model.feature_names_in_)
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,data):
        data[cat_cols]=data[cat_cols].astype('category')
        data.drop(['gender','PhoneService'],inplace=True)
        for i in bin_col:
            data[i]=le.fit_transform(data[i])
        df=pd.get_dummies(data=data,columns=multi_col,drop_first=True)
        





