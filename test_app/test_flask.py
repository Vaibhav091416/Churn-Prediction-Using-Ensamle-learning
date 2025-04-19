from app import app 
import json
import pandas as pd


def test_home():
    resoponse=app.test_client().get('/')

    assert resoponse.status_code==200

def test_predict():
    response=app.test_client()

    with open('../artifact/Unique_vals.json','r') as file:
        data=json.load(file)
        col_fields = dict(zip(data.keys(), [value.split(',') for value in data.values()]))
    with open('../artifact/num_col.json','r') as file:
        lis=json.load(file)
        num_fields=list(lis)
        
    test_data=dict()

    for field in col_fields:
        test_data[field] = col_fields[field][0]  # Choose first option as test input
    for field in num_fields:
        test_data[field] = 50  # Use dummy numeric value

    test_data=pd.DataFrame(test_data,index=[0])


    response = response.post('/predictdata', data=test_data, follow_redirects=True)

    assert response.status_code == 200
    assert b'The prediction is' in response.data

