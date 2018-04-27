# Refined MakeoverMonday Tweet collector

#collect tweets with date inputs in yyyy-mm-dd format
def collectTweets(startdate,enddate):
  
    #import libraries
    import tweepy
    import pandas as pd
    from credentials import consumer_key,consumer_secret, access_token, access_token_secret
    
    consumer_key=consumer_key
    consumer_secret=consumer_secret
    access_token=access_token
    access_token_secret=access_token_secret
    auth=tweepy.AppAuthHandler(consumer_key, consumer_secret)
    api=tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
    
    
    #pull tweets from twitter
    tweetjsons=[]
    for tweet in tweepy.Cursor(api.search,
                               q="#makeovermonday ",since=startdate, until=enddate,
                               include_entities=True,result_type='recent'
                               ,exlude_replies=True).items():
        if 'RT' not in tweet.text:#remove retweets
            tweetjsons.append(tweet)
            
    viz_info=[]
    for tweet in tweetjsons:
        ids=tweet.id
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
        text=tweet.text
        viz_info.append((ids,name,screen_name,date,tweet_url,text,location,time_zone,language,pic))
        
    df=pd.DataFrame(viz_info, columns=['ID','Name','TwitterHandle', 'Date', 'TweetUrl','Text','Location',
                                       'TimeZone','Language'])
    
    return df

#geocode locations into latitude and longitude columns in dataframe.
def GeocodeDF(df):
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
    df['Longitude']=long
    df[['Latitude','Longitude']].astype('float')
    
    df=df.drop(['Coordinates'],axis=1)
    
    return df

#import data csv, concat with new data frame, then export to csv
def unionExport(df):
    import os
    import pandas as pd
    os.chdir('D:\\MakeoverMondayDataFiles')
    data=pd.read_csv('data.csv',encoding='latin1')
    data=pd.concat([df,data], axis=0)
    data['Date'] = data['Date'].astype('str')
    data['Date']=pd.to_datetime(data['Date'])
    data.to_csv('data.csv',index=False)
    


