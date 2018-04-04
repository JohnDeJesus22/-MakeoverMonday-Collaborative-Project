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
data=pd.read_csv('data.csv',encoding='latin1')# to apply model on
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
y=model_data.loc[:,'RealorFake'].values


#tfidf vectorizer prep
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
X=cv.fit_transform(corpus).toarray()
y=model_data.loc[:,'RealorFake'].values

#hashingvectorizer prep
from sklearn.feature_extraction.text import HashingVectorizer
cv=HashingVectorizer(n_features=20)
X=cv.transform(corpus).toarray()#wrap in np.abs to convert to positive for multinomial
y=model_data.loc[:,'RealorFake'].values

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
#############################################################################################

#SVM (linear separatable version)
#Feature Scaling: for more accurate predictions
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Fitting SVM to Training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)


#k-fold cross validation for SVM 
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train,y=y_train,n_jobs=1,cv=10)
mean=accuracies.mean() #check accuracy(bias)
variance=accuracies.std()#determine variance
#4/2/18: kernel svm with tfidf vectorizor yielded about 55% accuracy....

########################################################################################

#4/3/18 attempting logistic regression since this is a binary classification

#Feature Scaling: for more accurate predictions
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Fitting Logistic Regression to the Training Set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuray=accuracy_score(y_test,y_pred)

#k-fold cross validation for Logistric Regression
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train,y=y_train,n_jobs=1,cv=10)
mean=accuracies.mean() #check accuracy(bias)
variance=accuracies.std()#determine variance
#4/3/18 with Count Vectorizer still ranging at about 51% accuracy...
#confusion matrix showed equal number of false positives and false negatives (41 each)

###########################################################################################
#4/3/18 attempting random forests since it is an ensemble method since kernel SVM was more
#successful

#Feature Scaling: for more accurate predictions
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Fitting Random Forest Classification to Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

#predicting the test results
y_pred=classifier.predict(X_test)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuray=accuracy_score(y_test,y_pred)

#k-fold cross validation for Random Forests
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train,y=y_train,n_jobs=1,cv=10)
mean=accuracies.mean() #check accuracy(bias)
variance=accuracies.std()#determine variance
#4/3/18 with Count Vectorizer: lower variance but accuracy was about 49%
#Tfidf Vectorizer: even lower variance but accuracy was still at about 49%
#Hashing Vectorizer: larger variance than previous two vectorizers and about the same accuracy

########################################################################################

#decided to give adaboost a try with decision trees
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
boost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),n_estimators=600,
                         learning_rate=1)
boost.fit(X_train,y_train)

y_pred=boost.predict(X_test)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuray=accuracy_score(y_test,y_pred)

#k-fold cross validation for Random Forests
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=boost, X=X_train,y=y_train,n_jobs=1,cv=10)
mean=accuracies.mean() #check accuracy(bias)
variance=accuracies.std()#determine variance
#4/3/18 no improvement versus the other classifiers. Perhaps I need more data, or should just
#change the bot to retweet all real tweets with the hashtag. Problem with that is it will include
#non-participant submissions or 'retweets' from them.
#Data.world has the treads for these already. Perhaps I should just take the images from there
#and retweet those. A bot similar to the 100days of code can still be made. Will need to rework
#the data set.