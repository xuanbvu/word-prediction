import tweepy
from basicTf import predict

# read credentials from file
with open("./auth.txt") as f:
    content = f.readlines()
    content = [x.strip() for x in content]

# authenticate to twitter
auth = tweepy.OAuthHandler(content[0], content[1])
auth.set_access_token(content[2], content[3])

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def tweet(prediction):
    l = prediction.split("\n")
    print(l)
    # api.update_status(l[0])


prediction = predict(u"hello")
tweet(prediction)
# api.update_status("she's not historical thanksgiving")
