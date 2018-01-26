#Reference Table of MOM participants and their locations

#Import libraries
import pandas as pd
import numpy as np

#Import data
data=pd.read_csv('data.csv',encoding='latin1')

#Create coordinates column
data['Coordinates']=[(data['Latitude'][i],data['Longitude'][i]) for i in range(len(data))]

#Creating Geography Reference Dataset
geo_ref=data[['Name','Latitude','Longitude']]
geo_ref=geo_ref.drop_duplicates('Name',keep='first')
geo_ref=geo_ref.dropna(how='any',axis=0)

#Export to csv
geo_ref.to_csv('MOM_Geo_Ref.csv', index=False)


