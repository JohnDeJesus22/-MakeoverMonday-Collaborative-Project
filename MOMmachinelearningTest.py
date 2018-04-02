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

######################################################################################
#setting up tweets for train/test split
#initialize corpus
corpus=[]

#create instances of tweettokenizer, detokenizer, and porterstemmer.
#initialize pattern to filter urls
twtoken=TweetTokenizer()
detokenizer=MosesDetokenizer()
ps=PorterStemmer()
pattern=re.compile(r'https\S+')

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
#set up Naive Bayes classifier due to satisfied assumptions
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
accuray=accuracy_score(y_test,y_pred)#75% on first test on 3/31/18(edit 4/1 outlier)

#k-fold cross validation for Naive Bayes (shows 50% twice on 4/1/18)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train,y=y_train,n_jobs=1,cv=10)
mean=accuracies.mean() #check accuracy(bias)
variance=accuracies.std()#determine variance
#following done after switching pattern to r'http\S+' instead of https
#4/1/18 first k-fold
#mean=48%
#variance=.05718
#so there is an overall low accuracy and med variance

#second k-fold (shuffled data again beforehand)
#mean=50.9%
#variance=.04882, lower variance than before

#third k-fold (shuffled data beforehand)
#mean=49%
#variance=.054721, worst then second k-fold
############################################################################################
#Fitting MultiNomial Naive Bayes to Training set (should have used this first since
#it is mainly for text classification)
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuray=accuracy_score(y_test,y_pred)#75% on first test on 3/31/18(edit 4/1 outlier)

#k-fold cross validation for Naive Bayes (shows 50% twice on 4/1/18)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train,y=y_train,n_jobs=1,cv=10)
mean=accuracies.mean() #check accuracy(bias)
variance=accuracies.std()#determine variance
#mean=48%
#variance=.05722, not much improvement


'''
Next Steps:
1. Determine best way to prep data via:
    a) bag of words (done with 2 naive bayes classifiers)
    b) td-idf
    c) word2vec
    d) others?
2. Visualize?
3. Determine best classifier (did naive bayes once due to satisfied assumptions)
4. Apply to mass data set.

'''


