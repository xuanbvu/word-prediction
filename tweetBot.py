import tweepy

# read credentials from file
with open("./auth.txt") as f:
    content = f.readlines()
    content = [x.strip() for x in content]

# authenticat to twitter
auth = tweepy.OAuthHandler(content[0], content[1])
auth.set_access_token(content[2], content[3])

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

api.update_status("test tweet from tweepy python")
