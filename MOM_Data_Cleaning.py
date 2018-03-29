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
false_to_data=['Paula Jennings','TD?']
for name in false_to_data:
    data,falsetweets=update_and_filter(name,data,falsetweets)
#######################################################################################    

#investigate the tweets of a single tweeter
check=data[data.Name=='Mark Bradbourne']

#discovered duplicates in urls, dropped 150 duplicate rows
data=data.drop_duplicates(subset='TweetUrl',keep='last')

false_urls=['https://t.co/m9evwhaVGA','https://t.co/I5PS74wVkg','https://t.co/5uHvtK566V',
            'https://t.co/c2UPPuhfYV','https://t.co/L0g5rIyE1x','https://t.co/bRaYmZ1LSG',
            'https://t.co/R2XFYgMtUt','https://t.co/3vwMm6aRJE','https://t.co/O4LMrZUDMg',
            'https://t.co/DnCBVXbrwI','https://t.co/tMCKbmGwTe','https://t.co/RSjQ4RyfiU',
            'https://t.co/qFN1pzl03f','https://t.co/aro9UmokbN','https://t.co/xDbGF2URE4',
            'https://t.co/4i1oj2KLs7','https://t.co/ahIKxEqYm3','https://t.co/P5dscVxYs9',
            'https://t.co/sNEQ6sBcFe','https://t.co/PsWqlt5IZ9','https://t.co/BLxYo6GVwp',
            'https://t.co/S0cXFthmy6','https://t.co/39oBzA7uce','https://t.co/ZYkRwD1Iak',
            'https://t.co/7JafWdKB6t','https://t.co/BJNU4Usqzh','https://t.co/BZE0qmynHb',
            'https://t.co/qPfi2gZM95','https://t.co/9lWNGV1uwD','https://t.co/QPoYVN7EaK',
            'https://t.co/aFGTk1bF1L','https://t.co/3O5C1bLxGd','https://t.co/d3IoWnRIsL',
            'https://t.co/aHl183nH2t','https://t.co/ZYlv1jyLcM','https://t.co/HhJnKntwFp',
            'https://t.co/dOUcrMUywI','https://t.co/aLNOffVlt1','https://t.co/G70h9BINX8',
            'https://t.co/1zBBTFANNY','https://t.co/2FLqT1jvNc','https://t.co/vPoz3BNSJu',
            'https://t.co/IZGf5t8VQi','https://t.co/ZRi7Dxa7dw','https://t.co/phxxknDOmY',
            'https://t.co/CNqn4fRmzJ','https://t.co/U2zdmEEMaY','https://t.co/iM6JKlPfps',
            'https://t.co/8K3iQR9Z6Z','https://t.co/byc5NV9M0u','https://t.co/RYChk6lLNs',
            'https://t.co/ZCHgoZsI9P','https://t.co/Uh7bG7sAjJ']

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
#update csvs
data.to_csv('data.csv',index=False)
falsetweets.to_csv('falsetweets.csv',index=False)
