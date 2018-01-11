#Twitter Extraction Code
#Used with Spyder 

#import libraries
import tweepy
import pandas as pd

#Keys to initiate connection to twitter api
consumer_key=	'comsumer key'
consumer_secret='consumer secret'
access_token='access token'
access_token_secret='access token secret'
auth=tweepy.AppAuthHandler(consumer_key, consumer_secret)
api=tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

#pulling tweets into a list
tweets=[]
for tweet in tweepy.Cursor(api.search,
                           q="#makeovermonday ",since='2018-01-01', until='2018-01-09',
                           include_entities=True,result_type='mixed'
                           ,exlude_replies=True).items():
    if 'RT' not in tweet.text:#remove retweets
        tweets.append(tweet)
        
#Loop for getting all the necessary info and store it in a list.
viz_info=[]
for tweet in test:
    name=tweet.author.name
    screen_name=tweet.author.screen_name
    try:
        location=tweet.author.location
    except:
        location='nan'
    try:
        time_zone=tweet.author.time_zone
    except:
        time_zone='nan'
    language=tweet.lang
    date=tweet.created_at
    try:
        tweet_url=tweet.entities['urls'][0]['url']
    except:
        tweet_url='NoLink'
    try:
        pic=tweet.entities['media'][0]['media_url']
    except:
        pic='nan'
    text=tweet.text
    viz_info.append((name,screen_name,date,tweet_url,text,location,time_zone,language,pic))
    
#import to viz info to dataframe, clean, and export to csv
df=pd.DataFrame(viz_info, columns=['Name','TwitterHandle', 'Date', 'TweetUrl','Text','Location','TimeZone','Language','VizPic'])

df=df[(df.Name !='Eva Murray')]#remove Andy and Eva tweets
df=df[df.Name !='Andy Kriebel']

df= df.drop(df['Text'].str.contains('#makeup'),axis=0)#drop makeup tweets
df=df[(df.TweetUrl!='NoLink') | (df.VizPic!='nan')]#removed tweets with no link or media url

#Filters for false tweets. Of course this is far from the best method but this is the first round.
#Model will be built to filter out irrevelant tweets
df=df[df['Text'].str.contains('home')==False]
df=df[df['Text'].str.contains('makeup')==False]
df=df[df['Text'].str.contains('designer')==False]
df=df[df['Text'].str.contains('dentist')==False]
df=df[df['Text'].str.contains('beauty')==False]
df=df[df['Text'].str.contains('granite')==False]
df=df[df['Text'].str.contains('blind')==False]
df=df[df['Text'].str.contains('teeth')==False]
df=df[df['Text'].str.contains('Vintage')==False]
df=df[df['Text'].str.contains('skin')==False]
df=df[df['Text'].str.contains('hair')==False]
df=df[df['Text'].str.contains('Paige')==False]
df=df[df['Text'].str.contains('Pool')==False]
df=df[df['Text'].str.contains('transformation')==False]
df=df[df['Text'].str.contains('house')==False]
df=df[df['Text'].str.contains('DIY')==False]
df=df[df['Text'].str.contains('Liz')==False]
df=df[df['Text'].str.contains('fitness')==False]
df=df[df['Text'].str.contains('kitchen')==False]
df=df[df['Text'].str.contains('quilt')==False]
df=df[df['Text'].str.contains('Paint')==False]
df=df[df['Text'].str.contains('score')==False]
df=df[df['Text'].str.contains('wheel')==False]
df=df[df['Text'].str.contains('deserves')==False]

df['Location'].str.strip('currently:')#remove 'currently:'
df['Location'].str.replace('/',',')#replace '/' with ','

df.to_csv('VizTweetData2018_01_09GMT.csv', index=False)