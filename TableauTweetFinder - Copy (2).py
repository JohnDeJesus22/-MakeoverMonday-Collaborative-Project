#Tableau Tweet Finder for MakeoverMonday

#import libraries
import tweepy
import pandas as pd
#import gspread
import time
#from oauth2client.service_account import ServiceAccountCredentials

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

#Keys to initiate connection to google api
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
credentials= ServiceAccountCredentials.from_json_keyfile_name('Makeover Monday-c88d3f2e7f8d.json', scope)
gc = gspread.authorize(credentials)

#pulling tweets in to a list #edited on 1/4/18 see copy for original search
tweetjsons=[]
for tweet in tweepy.Cursor(api.search,
                           q="#makeovermonday ",since='2018-04-24', until='2018-05-03',
                           include_entities=True,result_type='recent'
                           ,exlude_replies=True).items():
    if 'RT' not in tweet.text:#remove retweets
        tweetjsons.append(tweet)

#Loop for getting all the info
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

#import to dataframe, clean, and export to csv
df=pd.DataFrame(viz_info, columns=columns)

import os
os.chdir('D:\\MakeoverMondayDataFiles')
falsetweets=pd.read_csv('NewFalseTweets.csv',encoding='latin1')
real_tweets=pd.read_csv('NewRealTweets.csv',encoding='latin1')

Words=['photo','views','presentation','gallery','missing','home','makeup','dentist','beauty','granite','blind','teeth','Vintage','skin','hair','Paige','Pool','transformation',
       'house','DIY','Liz','fitness','kitchen','quilt','Paint','score','wheel','deserves',
       'mama','Yarn','#rich','VEGAN','Soak','#BubbleBathLovers','lash','room',
       'Wallpaper','refurbishment','chic','tub','Blinds','Butt','blog','odor','Serenity',
       '#Acne','Spa','Seattle','Treat','Pipes','commerical','#humor','#rapper','bubbles',
       '#RiseAndGrind','girl','face','Walls','Chanterelle','segment','powder','eyes','Office',
       'fat','office','entertainment','LIVE','#smile','Granite','facility','#Eagles','pantry',
       '#MondayMorning','Kait','facilities','extensions','Digest','code','recording']

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

df=df.drop_duplicates(subset='Text',keep='last')

for word in Words:
    x=df[df['Text'].str.contains(word)==True]
    falsetweets=falsetweets.append(x,ignore_index=True)

for fake in FakeTweeters:
    x=df[df.TwitterHandle ==fake]
    falsetweets=falsetweets.append(x,ignore_index=True)
    
#drop from df
for word in Words:
    df=df[df['Text'].str.contains(word)==False]
    
for fake in FakeTweeters:
    df=df[df.TwitterHandle!=fake]
      
y=df[(df.TweetUrl=='NoLink') & (df.VizPic==None)]
falsetweets=falsetweets.append(y,ignore_index=True)

df=df[(df.TweetUrl!='NoLink') & (df.VizPic!=None)]#removed tweets with no link and no media url

############################################################################################
#testing using gspread
wks = gc.open("pp 17").sheet1
wks.update_acell('B2', "it's down there somewhere, let me take another look.")
wks.update_acell('B1',"Glad I finally got this to work")
cell_list = wks.range('A1:B7')


testsheet=gc.create('TweetPullTest2')#created 1/2/17
testsheet=gc.open('TweetPullTest2').sheet1
testsheet=gc.open('TweetPullTest2')
testsheet.share('j.dejesus22@gmail.com',perm_type='user', role='writer')

testsheet.add_worksheet("insidesheet",10,10)# created a sheet inside the file 
#with 10 rows and columns

testsheet.list_permissions()#lists permisions on spreadsheet and their roles

testsheet.worksheets()#returns list of worksheets
test1=testsheet.sheet1#gave us the second sheet ('insidesheet') unless you open the sheet with 
#.sheet1 as was fixed above


#append rows to goggle sheet
for i in range(len(viz_info)):
    row=viz_info[i]
    testsheet.insert_row(row,i+1)

testsheet.share('robert@vizsimply.com', perm_type='user', role='reader',notify=True, 
                email_message='Hey Rob! Confirm if you got this and looked at it. I am sharing this from python.Another step closer ;)')
#can only share spreadsheets, not worksheets....
#have to reopen spreadsheet without .sheet1 method
######################################################################################
count=df['Location'].value_counts().sort_values(ascending=False)

#cleaning
df['Location']=df['Location'].str.replace('/',',')#replace '/' with ','
df['Location']=df['Location'].str.replace('via',',')#replace '/' with ','
df['Location']=df['Location'].str.replace('iPhone: 51.494932,-0.127700','Westminster, London')
df['Location']=df['Location'].str.replace('currently: AKL,NZ','AKL, NZ')
df['Location']=df['Location'].str.replace('YYZ','Toronto, Canada')


##############################################################################################
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def do_geocode(address):
    geopy = Nominatim()
    try:
        return geopy.geocode(address,exactly_one=True)
    except GeocoderTimedOut:
        return do_geocode(address)
    
df['Coordinates']=df['Location'].apply(lambda x: do_geocode(x) if x != None else None)

lat=[]
long=[]
for i in df['Coordinates']:
    if i== None:
        lat.append(None)
        long.append(None)
    else:
        lat.append(i.latitude)
        long.append(i.longitude)
df['Latitude']=lat
df['Latitude'].astype('float')
df['Longitude']=long
df['Longitude'].astype('float')

df=df.drop(['Coordinates'],axis=1)


real_tweets=real_tweets.append(df,ignore_index=True)

#done to fix inconsistent date issue
data=pd.read_csv('data.csv',encoding='latin1')
data=pd.concat([df,data], axis=0)
data['Date'] = data['Date'].astype('str')
data['Date']=pd.to_datetime(data['Date'])


data.to_csv('data.csv',index=False)
falsetweets.to_csv('NewFalseTweets.csv', index=False)
real_tweets.to_csv('NewRealTweets.csv',index=False)
