import pandas as pd
import numpy as np
import os
import re

#change directory and get data
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

#create dataframe for 258 real tweets
real_tweets=pd.DataFrame([],columns=data.columns.tolist())

#pull keys
keys=list(participant_dict.keys())

#populate data frame
for i in keys:
    if i>=15:
        for name in participant_dict[i]:
            real_tweets=real_tweets.append(data[data.Name==name],ignore_index=True)

#drop lat and long
real_tweets=real_tweets.drop(['Latitude','Longitude'],axis=1)
falsetweets=falsetweets.drop(['Latitude','Longitude'],axis=1)

#create labels for classification
real_tweets['RealorFake']=1
falsetweets['RealorFake']=0

#combine real and false tweets
model_data=pd.concat([real_tweets,falsetweets],axis=0)

'''
Next Steps:
1. Pull and clean text
2. Determine best we to change data (i.e. use bag of words, tfidf, word2vec)
3. Visualize
4. Determine best classifier 
5. Apply to mass data set and to tomorrow's pulled data.

'''


