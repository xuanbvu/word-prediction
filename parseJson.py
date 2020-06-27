import json
import os

all = open("./fb.txt", "a")

for filename in os.listdir("messages"):
    with open("./messages/{}".format(filename), "r") as f:
        data = json.load(f)
    for m in data["messages"]:
        if m["sender_name"] == "Xuan Vu":
            if ("content" in m):
                # don't add links
                if "https" not in m["content"]:
                    print(m["content"])
                    all.write(m["content"]+"\n")

all.close()
