# NLP with #makeovermonday Tweets (in progress)

#import libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

#import data
data=pd.read_csv('data.csv',encoding='latin1')


#isolate tweets
tweets=data['Text']

#Initially TweetTokenizer
twtoken=TweetTokenizer()

for tweet in data['Text']:
    tweet.strip('http')

#clean tweets
corpus=[]
for i in range(data.shape[0]):
    posts=re.sub('[^a-zA-z#]',' ',data['Text'][i])
    posts=posts.lower()
    posts=twtoken.tokenize(posts)
    ps=PorterStemmer()
    posts=[ps.stem(word) for word in posts if not word in set(stopwords.words('english'))]
    posts=' '.join(posts)
    corpus.append(posts)

#setting up bag of words model by creating count matrix
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()


#need to label these tweets and false tweets to build classifier.