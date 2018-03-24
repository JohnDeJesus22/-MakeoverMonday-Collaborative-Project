# MakeoverMonday  Bot Testing

#import libraries
import tweepy
import pandas as pd
import time
import os
os.chdir('path to scripts')
from RefinedMOMTweetRetrieverCode import collectTweets
from credentials import *

#Keys to initiate connection to twitter api for bot
consumer_key=	consumer_key
consumer_secret=consumer_secret
access_token=access_token
access_token_secret=access_token_secret
auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

#pull tweets
tweetjsons=[]
for tweet in tweepy.Cursor(api.search,q='#makeovermonday',since='2018-03-17', until='2018-03-24',
                           include_entities=True,result_type='recent'
                           ,exlude_replies=True).items():
    if 'RT' not in tweet.text:
        tweetjsons.append(tweet)

#words contained by false tweets        
Words=['makeup','dentist','beauty','granite','blind','teeth','Vintage','skin','hair','Paige','Pool','transformation',
       'house','DIY','Liz','fitness','kitchen','quilt','Paint','score','wheel','deserves',
       'mama','Yarn','#rich','VEGAN','Soak','#BubbleBathLovers','lash','room',
       'Wallpaper','refurbishment','chic','tub','Blinds','Butt','blog','odor','Serenity',
       '#Acne','Spa','Seattle','Treat','Pipes','commerical','#humor','#rapper','bubbles',
       '#RiseAndGrind','girl','face','Walls','Chanterelle','segment','powder','eyes','Office',
       'fat','office','entertainment','LIVE','#smile','Granite','facility','#Eagles','pantry',
       '#MondayMorning','Kait','facilities','extensions','Digest','code','recording']

#remove false tweets based on words
for tweet in tweetjsons:
    for word in Words:
        if word in tweet.text:
            tweetjsons.remove(tweet)

#eva, andy and unrelated tweeters       
Non_participants=['TriMyData','VizWizBI','bronzenblush','thedailystarr','ColorCraftsmen',
              'RockvilleTownSq','HollyBurnard','aglahet_','WarriorofGod97',
              'makeupbyjanneth','ASalonForHair','projectmakeover','love_sparklezz',
              'sheldenarch','sheldenarch','PussyandPooch','newjerseykelly','FoodTronic',
              'theTahirWoods','TiffanyTopie','TheRockRMS','IntlDesignGuild','allinormskirk',
              'nshpatel69','BOBOULESZ']

#remove tweets made by above list      
for tweet in tweetjsons:
    for twitterhandle in Non_participants:
        if tweet.author.screen_name==twitterhandle:
            tweetjsons.remove(tweet)

#retweet remaining tweets
for tweet in tweetjsons:
    api.retweet(tweet.id)