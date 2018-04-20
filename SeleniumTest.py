# Selenium Data.world scraping

#import libraries
import time
import re
import pandas as pd
import selenium
from selenium import webdriver

start_time=time.time()
#logging in to data.world
driver = webdriver.Chrome('chromedriver.exe')
driver.get('https://data.world/login')
username = driver.find_element_by_xpath("//input[@placeholder='Username or email address']")
password = driver.find_element_by_xpath("//input[@placeholder='Password']")

#input username and password
username.send_keys("username")
password.send_keys("password")
driver.find_element_by_xpath("//button[@type='submit']").click()

#list of links
week_urls=['https://data.world/makeovermonday/2018-w-1-u-s-per-capita-consumption-of-poultry-livestock/discuss/2018-w1-u-s-per-capita-consumption-of-poultry-livestock/69848',
            'https://data.world/makeovermonday/2018-w-2-looks-vs-personality/discuss/2018-w2-looks-vs-personality/71005',
            'https://data.world/makeovermonday/2018-w-3-u-s-household-income-distribution-by-state/discuss/2018-w3-u-s-household-income-distribution-by-state/72874',
            'https://data.world/makeovermonday/2018-w-4-turkey-vulture-migration-in-north-and-south-america/discuss/2018-w4-turkey-vulture-migration-in-north-and-south-america/74472',
            'https://data.world/makeovermonday/2018-w-5-what-the-most-profitable-companies-make-per-second/discuss/2018-w5-what-the-most-profitable-companies-make-per-second/76138',
            'https://data.world/makeovermonday/2018-w-6-baseball-demographics-1947-2016/discuss/2018-w6-baseball-demographics-1947-2016/78179',
            'https://data.world/makeovermonday/2018w7-the-winter-olympics/discuss/2018-w7-the-winter-olympics/80806',
            'https://data.world/makeovermonday/2018w8-where-does-your-medicine-come-from/discuss/2018-w8-where-does-your-medicine-come-from/82935',
            'https://data.world/makeovermonday/2018w9-world-economic-freedom/discuss/2018-w9-world-economic-freedom/85177',
            'https://data.world/makeovermonday/w102018-what-policymakers-know-about-women-and-girl-issues/discuss/2018-w10-what-policymakers-know-about-women-and-girl-issues/86558',
            'https://data.world/makeovermonday/2018w11-growth-in-irish-whiskey-sales/discuss/2018-w11-growth-in-irish-whiskey-sales/90031',
            'https://data.world/makeovermonday/2018w12-uk-pet-population-in-2017/discuss/2018-w12-uk-pet-population-in-2017/92656',
            'https://data.world/makeovermonday/what-is-the-uks-favorite-chocolate-bar/discuss/2018-w13-what-is-the-uks-favorite-chocolate-bar/95666',
            'https://data.world/makeovermonday/2018w14-world-wine-production/discuss/2018-w14-world-wine-production/98310',
            'https://data.world/makeovermonday/2018w15-arctic-sea-ice-extent/discuss/2018-w15-arctic-sea-ice-extent/100922'
            ]


#go to url
time.sleep(5)
info=[]
pattern=re.compile(r'https://data.world/S+')
for week_num,week_url in enumerate(week_urls):
    driver.get(week_url)
#get post
    posts=driver.find_elements_by_xpath("//div[@class='post__thumbnail___33jZY DatasetCard__thumbnail___33EHZ']")
    for i in range(1,len(posts)):
        tag_name = posts[i].find_element_by_xpath(".//a[@class='post__username___YIalv']")
        dw_name=tag_name.text[1:]
        try:
            vizpic=posts[i].find_element_by_xpath(".//img")
            vizpic=vizpic.get_attribute('src')
        except:
            vizpic=None
        try:
            #first a leads to link to profile. Will be useful to get real names
            #link=posts[i].find_element_by_xpath('.//a').get_attribute('href')
            #will use above with driver.execute_script("window.history.go(-1)")
            #to return to original page after getting names from profile. (snippet from stackoverflow)
            link=posts[i].find_element_by_xpath('.//a[@target="_blank"]').get_attribute('href')
            '''
            need to adjust below while loop
            tag_string=1
            while re.match(pattern,link)==True:
                link=posts[i].find_element_by_xpath('.//a[@target="_blank"]'+str([tag_string])).get_attribute('href')
                tag_string+=1
            '''
        except:
            link=None
        info.append((dw_name, vizpic,link,week_num+1))

#create data frame
data=pd.DataFrame(info,columns=['DwName','VizPic', 'Link','WeekNumber'])

#scrap real names of participants based on data.world username
real_names=[]
profile_pics=[]
DwName_unique=data['DwName'].unique().tolist()
for name in DwName_unique:
    #go to webpage
    driver.get('https://data.world/'+ name)
    #get name and append to list
    real_name=driver.find_element_by_xpath("//span[@class='dw-name UserCard__displayname___1s6YY']")
    real_name=real_name.text
    real_names.append(real_name)
    
#create dictionary to map on to new columns
dw_to_real_dict=dict(zip(DwName_unique,real_names))
data['RealName']=data['DwName'].map(dw_to_real_dict)

#signout and close the window
time.sleep(3)
driver.find_element_by_xpath("//li[@class='Header__menu_account___3jT6C dw-UserSettingsDropdown dropdown']").click()
time.sleep(5)
driver.find_element_by_xpath("//li[@class='Header__itemSignOut___UfV3_']").click()
time.sleep(5)
driver.quit()

print(time.time()-start_time)
#export csv 
#data.to_csv('datadotworld_full_scrap_attempt.csv',index=False)

#xpath to a link after mentions of other users in forms of (@tags)
#//*[@id="110696"]/div/div/div/span/p[5]/a