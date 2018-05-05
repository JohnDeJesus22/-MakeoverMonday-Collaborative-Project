# FalseTweet Collecting

import tweepy
import pandas as pd


#Keys to initiate connection to twitter api
consumer_key=	''
consumer_secret=''
access_token=''
access_token_secret=''
auth=tweepy.AppAuthHandler(consumer_key, consumer_secret)#changed from OAuth to AppAuth
#for better rate. AppAuth doesn't need auth.set_access_token, OAuth does.
#auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token,access_token_secret)#
api=tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

FakeTweeters=['Mayrankin36','AvasMomTeaches','CuttingRoomCre8','mambuzz13','PRBdata',
              'theinfolabie','L_Coull','GraceDivineArt','fabrichut67','andrewvelazcom',
              'TRetailFloors','FBCT_Voice','Coffeebreakexec','JohnNewmanHair',
              'adventresults','razactually','GraceDivineArt','NWdermatology',
              'stltug','continuus_tech','Caninetofive','ULTIMATESHAVE1','farhatasha',
              'AmeliaPisano6ix','Y94','bronzenblush','thedailystarr','ColorCraftsmen',
              'RockvilleTownSq','HollyBurnard','aglahet_','WarriorofGod97',
              'makeupbyjanneth','ASalonForHair','projectmakeover','love_sparklezz',
              'sheldenarch','sheldenarch','PussyandPooch','newjerseykelly','FoodTronic',
              'theTahirWoods','TiffanyTopie','TheRockRMS','IntlDesignGuild','allinormskirk',
              'nshpatel69','BOBOULESZ','winggirlmethod','yeouth','rezyarianda','EdenVision',
              'sunset','paintshopcanada','caddy_face','SpruceAtHome','JingeRSays2','decorlasting',
              'PortmanDanielle','decorlasting','Rock_N_Veg','YolandaRWriter','LED_Curator',
              'PowerPlumbingCH','Mingham','travelitalian1','jcg_gabriz20','HirushaSupun',
              'TheFaceExpert','CatastrophBlo','redefinefinance','JakubWasTaken','immeg92'
              'shan10111','TMEXICANBEAUTY','smudgechicago','SqueezingsComix','BellissimaElle_',
              'BMCA_volunteer','shopwithsheray','TumiAmina','tinkerama','gloritere']


tweetjsons=[]
#for tweeter in FakeTweeters:
for screen_name in FakeTweeters:
    for tweet in tweepy.Cursor(api.user_timeline, screen_name=screen_name,
                           q="#makeovermonday ",since='2018-05-05', until='2018-05-05',
                           include_entities=True,result_type='recent',
                           exclude_replies=True).items():
        if 'RT' not in tweet.text:#remove retweets
            tweetjsons.append(tweet)
    

#create data frame
viz_info=[]
for tweet in tweetjsons:
    name=tweet.author.name
    screen_name=tweet.author.screen_name
    try:
        location=tweet.author.location
    except:
        location=None
    try:
        time_zone=tweet.author.time_zone
    except:
        time_zone=None
    language=tweet.lang
    date=tweet.created_at
    try:
        tweet_url=tweet.entities['urls'][0]['url']
    except:
        tweet_url='NoLink'
    try:
        pic=tweet.entities['media'][0]['media_url']
    except:
        pic=None
    text=tweet.text
    viz_info.append((name,screen_name,date,tweet_url,text,location,time_zone,language,pic))

columns=['Name','TwitterHandle', 'Date', 'TweetUrl', 'Text','Location','TimeZone','Language','VizPic']

df=pd.DataFrame(viz_info, columns=columns)

import os
os.chdir('D:\\MakeoverMondayDataFiles')
falsetweets=pd.read_csv('NewFalseTweets.csv',encoding='latin1')

falsetweets=falsetweets.append(df,ignore_index=True)

falsetweets.to_csv('NewFalseTweets.csv', index=False)