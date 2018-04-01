import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords #to use stopwords function and list
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize.moses import MosesDetokenizer #had to nltk download 'perluniprops' first

#change directory and get data
os.chdir('D:\\MakeoverMondayDataFiles')
data=pd.read_csv('data.csv',encoding='latin1')
falsetweets=pd.read_csv('falsetweets.csv',encoding='latin1')
realtweets=pd.read_csv('realTweets.csv',encoding='latin1')

'''
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
'''
#########################################################################################
#Build training data
#create labels for classification
realtweets['RealorFake']=1
falsetweets['RealorFake']=0

#combine real and false tweets, drop lat and long, and set index
model_data=pd.concat([realtweets,falsetweets],axis=0)
model_data=model_data.drop(['Latitude','Longitude'],axis=1)
model_data=model_data.set_index([[i for i in range(model_data.shape[0])]])

#shuffle the data
from sklearn.utils import shuffle
model_data=shuffle(model_data)
######################################################################################

#testing text cleaning on a single tweet#######
test=model_data['Text'][1]

pattern=re.compile(r'https\S+')#remove urls
urls=re.findall(pattern,test)
twtoken=TweetTokenizer()
test=twtoken.tokenize(test)
for url in urls:
    if url in test:
        test.remove(url)

detokenizer=MosesDetokenizer()
test=detokenizer.detokenize(test,return_str=True)#return back to string

test=re.sub('[^a-zA-z]',' ',test)#remove extra characters and punctuation

test=test.lower()#turn all words into lowercase

test=test.split()#split again to remove stop words

ps=PorterStemmer()#remove stop words. need to adjust for non english
test=[ps.stem(word) for word in test if not word in set(stopwords.words('english'))]
test=' '.join(test)#refuse tweet
##################################################################################
#initialize corpus
corpus=[]

#create instances of tweettokenizer, detokenizer, and porterstemmer
twtoken=TweetTokenizer()
detokenizer=MosesDetokenizer()
ps=PorterStemmer()

#build corpus
for i in range(model_data.shape[0]):
    text=model_data['Text'][i]
    urls=re.findall(pattern,text)
    text=twtoken.tokenize(text)
    for url in urls:
        if url in text:
            text.remove(url)
    text=detokenizer.detokenize(text,return_str=True)
    text=re.sub('[^a-zA-z]',' ',text)
    text=text.lower()
    text=text.split()
    text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text=' '.join(text)
    corpus.append(text)
#############################################################################################
#testing with bag of words model and spliting data
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)# returns 1500 columns of most frequent words
X=cv.fit_transform(corpus).toarray()
y=model_data['RealorFake'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test=train_test_split(X,y, test_size=0.20, random_state=0)

######################################################################
#set up naive bayes classifier due to satisfied assumptions
#that is:
#binary classification
#tweets are independent of each other
#evidence probability is the same for all tweets

#Fitting Naive Bayes to Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#predicting the test results
y_pred=classifier.predict(X_test)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuray=accuracy_score(y_test,y_pred)#75% on first test on 3/31/18

'''
Next Steps:
1. Determine best way to prep data via:
    a) bag of words (done with one classifier)
    b) td-idf
    c) word2vec
    d) others?
2. Visualize?
3. Determine best classifier (did naive bayes once due to satisfied assumptions)
4. Apply to mass data set.

'''


