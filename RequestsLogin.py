#Testing Requests with a Login

import requests

#create instances for login url and desired url 
login_url='https://data.world/login'

desired_url='https://data.world/makeovermonday/what-is-the-uks-favorite-chocolate-bar\
                /discuss/2018-w13-what-is-the-uks-favorite-chocolate-bar/95666'
                
username='my_username'
password='my_password'
#create dictionary for appropriate login info            
credentials={'email':username,'password':password}

#access desired page and print text
with requests.Session() as session:
    post=session.post(login_url,data=credentials)
    r=requests.get(desired_url)
    print(r.text)
    


