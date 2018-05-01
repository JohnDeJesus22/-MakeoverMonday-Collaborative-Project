#Tweet Filter ML Script

class TweetClassifier:
    
    def fuse_and_label():
        #change directory and get data
        os.chdir('D:\\MakeoverMondayDataFiles')
        falsetweets=pd.read_csv('NewFalsetweets.csv',encoding='latin1')
        realtweets=pd.read_csv('NewRealTweets.csv',encoding='latin1')
        
        #create labels for classification
        realtweets['RealorFake']=1
        falsetweets['RealorFake']=0
        
        #combine real and false tweets, drop lat and long, and set index
        model_data=pd.concat([realtweets,falsetweets],axis=0)
        model_data=model_data.set_index([[i for i in range(model_data.shape[0])]])
        
        #shuffle the data
        from sklearn import shuffle
        model_data=shuffle(model_data,random_state=0)
        
        return model_data

    def build_corpus(model_data):
        #import libraries
        import nltk
        from nltk.tokenize import TweetTokenizer
        from nltk.corpus import stopwords #to use stopwords function and list
        from nltk.stem.porter import PorterStemmer 
        from nltk.tokenize.moses import MosesDetokenizer
        
        #setting up tweets for train/test split
        #initialize corpus
        corpus=[]
        
        #create instances of tweettokenizer, detokenizer, and porterstemmer.
        #initialize pattern to filter urls and usernames
        twtoken=TweetTokenizer()
        detokenizer=MosesDetokenizer()
        ps=PorterStemmer()
        url_pattern=re.compile(r'https\S+')
        user_pattern=re.compile(r'@\S+')
        
        #build corpus
        for i in range(model_data.shape[0]):
            text=model_data['Text'][i]
            urls=re.findall(url_pattern,text)
            users=re.findall(user_pattern,text)
            users=[re.sub('[^@a-zA-z]','',user) for user in users]
            text=twtoken.tokenize(text)
            for url in urls:
                if url in text:
                    text.remove(url)
            for user in users:
                if user in text:
                    text.remove(user)
            text=detokenizer.detokenize(text,return_str=True)
            text=re.sub('[^a-zA-z]',' ',text)
            text=text.lower()
            text=text.split()
            try:
                text.remove('makeovermonday')
            except Exception:
                pass
            text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
            text=' '.join(text)
            corpus.append(text)
        
        return corpus
    
    def Model_Prep(corpus,model_data):
        #testing with bag of words model with tfidftransformer and spliting data
        from sklearn.feature_extraction.text import CountVectorizer
        cv=CountVectorizer(max_features=1000,strip_accents='ascii')# returns 1000 columns of most frequent words
        X=cv.fit_transform(corpus).toarray()
        y=model_data.loc[:,'RealorFake'].values
        
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidf=TfidfTransformer()
        X=tfidf.fit_transform(X)
        
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train, y_test=train_test_split(X,y,
                                                        test_size=0.20, random_state=0)
        
        return X_train,X_test,y_train, y_test
        
        
    def TweetNB(X_train,y_train, X_test):
        from sklearn.naive_bayes import MultinomialNB
        classifier=MultinomialNB(alpha=.87,fit_prior=True)#setting fit-prior=false improved variance
        classifier.fit(X_train,y_train)
        
        y_pred=classifier.predict(X_test)
        
        return y_pred