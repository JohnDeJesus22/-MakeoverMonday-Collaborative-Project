#Tightied Code for Getting locations of MOM Tweeters.
#Code is steps toward automation.

#Import Libraries and Classes
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

#Load Data
df=pd.read_csv('data.csv',encoding='latin1')

#Modified Function found on stackoverflow to geocode locations to obtain lat
#and long. Done to override the timeout error.
def do_geocode(address):
    geopy = Nominatim()
    try:
        return geopy.geocode(address,exactly_one=True)
    except GeocoderTimedOut:
        return do_geocode(address)

#Creating Geocoded Location column
df['GeocodedLocation']=df['Location'].apply(lambda x: do_geocode(x) if x != None else None)

#Create the Latitude Column
lat=[]
for i in df['GeocodedLocation']:
    if i== None:
        lat.append(None)
    else:
        lat.append(i.latitude)
df['Latitude']=lat
df['Latitude'].astype('float')

#Create the Longitude Column
long=[]
for i in df['GeocodedLocation']:
    if i== None:
        long.append(None)
    else:
        long.append(i.longitude)
df['Longitude']=long
df['Longitude'].astype('float')

#Drop GeocodedLocation Column
df=df.drop(['GeocodedLocation'],axis=1)

#Export Data to a csv
df.to_csv('data.csv', index=False)
