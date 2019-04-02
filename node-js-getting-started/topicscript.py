#from firebase.firebase import FirebaseApplication, FirebaseAuthentication

#authentication = FirebaseAuthentication('AFH2c3nhbeT6gvRfjlg7iKHYROYVZmz4CIstByVy', 'em3700@yahoo.com', True, True)
#firebase = FirebaseApplication('https://breezytalk.firebaseio.com', authentication)

import os
import json
import time

def PUT(entry, table):
    os.system("curl -X PUT -d '" + json.dumps(entry) + "' '" + 'https://breezytalk.firebaseio.com' + table + "'")

topics = ["Religion", 
	"Science",
	"Politics",
	"Philosophy",
	"Education",
	"Food",
	"Sports",
	"Ethics",
	"Culture",
	"Music",
	"Hobbies",
	"Movies and Film",
	"Literature",
	"Comedy",
	"Something",
        "Anything"]
running = True
i = 0
while running == True:
	topic = topics[i]
	PUT(topic, '/topics/.json')
	i += 1
        if i > len(topics)-1:
		i = 0
	time.sleep(1)
