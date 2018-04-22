# Second test with ml model using fake tweets and #makeovermonday viz related tweets
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize.moses import MosesDetokenizer 

#change directory and get data
os.chdir('D:\\MakeoverMondayDataFiles')
data=pd.read_csv('data.csv',encoding='latin1')# to apply model on
falsetweets=pd.read_csv('falsetweets.csv',encoding='latin1')
realtweets=pd.read_csv('realTweets.csv',encoding='latin1')

#sending correct tweets back to realtweets csv
os.chdir('C:\\Users\\Sabal\\Desktop\\Data Science Class Files\\Random Practice\\MakeoverMondayProjectScripts')
from MOM_Data_Cleaning import update_and_filter

#create list of names to return correct tweets
names=['Eva Murray','Andy Kriebel','Sarah Burnett','John DeJesus','Tableau Software','STL Tableau User Grp',
       'Sarah Burnett','Tableau Public','Doc Kevin Lee Elder','EqualMeasures2030',
       'The Information Lab','Information Lab Irl','Infographics News','Data Science Renee',
       'Rodrigo Calloni','Simon Beaumont','Susan Glass','Sarah Bartlett','Jade Le Van',
       'Chantilly J','Jeff Plattner','Ken Flerlage','Umar Hassan','Mark Bradbourne',
       'Axel W.','Nish Goel','Neil Richards','Robert Crocker','federico lusi','Marc Soares',
       'Slalom','ClickInsight','FusionCharts','Dr. Trent Tucker','Curtis Harris',
       'Klaus Schulte','Charlie Hutcheson','Sean Miller','Rosario Gauna','Staticum',
       'Yoshihito Kimura','Louis Archer','Lindsay Betzendahl','Matt Wilson','Slalom Philadelphia',
       'Young Song','Joel Gluck','Paula Jennings','Laura Sutton','Pablo Gomez','Philip Riggs',
       'Christina P. Gorga','James Eselgroth','Fi Gordon','Mark Edwards','Ketan Deshpande',
       'Ann Jackson','Michael DiPalma','Sian Miller','Guillaume Martin','Brittany Hink','LM-7']

#append to names remaining MOM participants at the bottom of false tweets list
for index in falsetweets.index:
    if index > 454:
        names.append(falsetweets.get_value(index,'Name'))

#swap locations of data entries with names from falsetweets csv to realtweets csv
for name in names:
    realtweets,falsetweets=update_and_filter(name,realtweets,falsetweets)

#Create new csvs and save them
os.chdir('D:\\MakeoverMondayDataFiles')
falsetweets.to_csv('NewFalseTweets.csv',index=False)
realtweets.to_csv('NewRealTweets.csv',index=False)

#Build training data
#create labels for classification
realtweets['RealorFake']=1
falsetweets['RealorFake']=0

#combine real and false tweets, drop lat and long, and set index
model_data=pd.concat([realtweets,falsetweets],axis=0)
model_data=model_data.set_index([[i for i in range(model_data.shape[0])]])

#shuffle the data
from sklearn.utils import shuffle
model_data=shuffle(model_data)

