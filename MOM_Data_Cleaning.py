#MOM Data Cleaning

#import libraries
import pandas as pd
import numpy as np
import os
#import matplotlib.pyplot as plt
#import seaborn as sns
#import missingno as msno

#change directory and import dataset
os.chdir('D:\\MakeoverMondayDataFiles')
data=pd.read_csv('data.csv',encoding='latin1')
falsetweets=pd.read_csv('falsetweets.csv',encoding='latin1')
#####################################################################################
#Post frequency by name and create total posts lists
post_freq=data.Name.value_counts(ascending=False)
post_totals=sorted(post_freq.unique().tolist())

#Now to get a list of the indexes (aka names of 'participants')
participants_grouped=[]
for total in post_totals:
    participants_grouped.append(post_freq[post_freq==total].index.tolist())

#create dictionary
participant_dict=dict(zip(post_totals,participants_grouped))
#####################################################################################
#function for updating falsetweets and filtering out data
def update_and_filter(name,ft,df):
    false=ft.append(df[df.Name==name].iloc[:,:9],ignore_index=True)
    df=df[df.Name!=name]
    return false, df

#move two true tweets from falsetweets.csv to data.csv
false_to_data=['Slalom Philadelphia']
for name in false_to_data:
    data,falsetweets=update_and_filter(name,data,falsetweets)
    
#move rows from data to falsetweets
false_to_data=['Slalom Philadelphia']
for name in false_to_data:
    falsetweets,data=update_and_filter(name,falsetweets,data)
#######################################################################################    

#investigate the tweets of a single tweeter
check=data[data.Name=='Frederic Fery']


#discovered duplicates in urls, dropped 150 duplicate rows
data=data.drop_duplicates(subset='TweetUrl',keep='last')

false_urls=['https://t.co/WNVOU5SSFG','https://t.co/OzE8YHK0KV','https://t.co/DZAbWnBcN2',
            'https://t.co/bM6Xdm7sSM']

#filter out data by #tableauff
tableauff=data[data.Text.str.contains('#TableauFF')==True]
falsetweets=falsetweets.append(tableauff,ignore_index=True)
data=data[data.Text.str.contains('#TableauFF')==False]

                                      
#modified filter function for any column
def update_and_filter_by_column(column,item,ft,df):
    false=ft.append(df[df[column]==item].iloc[:,:9],ignore_index=True)
    df=df[df[column]!=item]
    return false, df

#filter out false_urls and place them in falsetweets
for url in false_urls:
    falsetweets,data=update_and_filter_by_column('TweetUrl',url,falsetweets,data)

data,falsetweets=update_and_filter_by_column('Name','chitraxi raj',data,falsetweets)

#unfortunately pulls relevant tweets with @andy and @eva twitterhandles
for string in data['Text']:
    if string[0]=='@' and (string[0:19]!='@VizWizBI @TriMyData' or string[0:19]!='@TriMyData @VizWizBI'):
        falsetweets,data=update_and_filter_by_column('Text',string,falsetweets,data)
#update csvs
data.to_csv('data.csv',index=False)
falsetweets.to_csv('falsetweets.csv',index=False)
