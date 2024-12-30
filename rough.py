import json 
with open('./artifact/Unique_vals.json','r') as file:
    data=json.load(file)
col_fields = dict(zip(data.keys(), [value.split(',') for value in data.values()]))
with open('./artifact/num_col.json','r') as file:
    lis=json.load(file)
num_fields=list(lis)

print(col_fields,'\n\n\n',num_fields)