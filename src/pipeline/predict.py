import pickle as pkl
import pandas as pd
import os 
import json


class PredictPipeline:

    def __init__(self):
        #Loading label encoder
        # print("initaiting object in pipeline")
        with open('./artifact/label_encoders.pkl','rb') as file_obj:
            self.le=pkl.load(file_obj)

        #Loading Categorical Columns
        with open('./artifact/Unique_vals.json','r') as file:
            data=json.load(file)

        #Loading multi, bin, cat_col
        self.col_fields = dict(zip(data.keys(), [value.split(',') for value in data.values()]))
        self.cat_cols=list(self.col_fields.keys())
        self.bin_col=[col for col in list(self.col_fields.keys()) if len(self.col_fields[col])==2]
        self.bin_col.remove('gender')
        self.bin_col.remove('PhoneService')
        self.multi_col=[col for col in list(self.col_fields.keys()) if len(self.col_fields[col])>2]
        self.multi_col=list(self.multi_col)
        self.multi_col = [col.strip() for col in self.multi_col]
        print(type(self.multi_col))

        #Loading Numerical Columns
        with open('./artifact/num_col.json','r') as file:
            lis=json.load(file)
        self.num_fields=list(lis)

        #Loading the standard sclaer
        with open('./artifact/scaler.pkl','rb') as file_obj:
            self.scaler=pkl.load(file_obj)

        #Loading drop list
        with open('./artifact/final_drop_list.json','r') as file_obj:
            drop_list=json.load(file_obj)
        self.drop_list=list(drop_list)

        #Loading the model
        with open('./artifact/final_model_abcl.pkl',"rb") as file_obj:
            self.model=pkl.load(file_obj)
        print("Model loaded ")
        with open('./artifact/after_dummies.json',"rb") as file_obj:
            self.added_dummies=json.load(file_obj)
        print("all artifacts loaded")
        self.model_feauture_list=list(self.model.feature_names_in_)
        self.model_feauture_list=list(self.model_feauture_list)



    def data_transform(self,data):

        data[self.cat_cols]=data[self.cat_cols].apply(lambda x:x.str.strip())
        data[self.cat_cols]=data[self.cat_cols].astype('category')
        data[self.num_fields]=data[self.num_fields].astype('float32')

        data.drop(['gender','PhoneService'],axis=1,inplace=True)
        print("gender and Phoneservices dropped.")
        for i in self.bin_col:
            data[i]=self.le[i].fit_transform(data[i])
        print("Label Encoding done.")

        data1=pd.DataFrame(0,index=[0],columns=self.model.feature_names_in_)
        for feat in self.model.feature_names_in_:
  
            if '_' in feat:
                x,y=feat.split('_')
                if data[x].iloc(0)==y:
                    data1[feat]=1

        data1['tenure']=data['tenure']
        data1['MonthlyCharges']=data['MonthlyCharges']
        data1['TotalCharges']=data['TotalCharges']

        

        print(data1)
        num_df=data1[self.num_fields]
        print("num_df: ",num_df)
        data_org=data1.copy()

        data1.drop(columns=self.num_fields,inplace=True,axis=1)

        scaled_num_df=self.scaler.fit_transform(num_df)
        scaled_num_df=pd.DataFrame(scaled_num_df,columns=self.num_fields)
        print("Scaled Numerical datframe: ",scaled_num_df)
        data1=data1.merge(scaled_num_df,left_index=True,right_index=True,how='left')
        print("transformed data: ",data1)
        return data1

    def predict(self,data):
        print('prediction begin')
  
        print(data)

        ans=self.model.predict(data)
        return ans