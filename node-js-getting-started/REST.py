#!usr/bin/env python

def POST(self, entry, table):
    os.system("curl -X POST -d '" + json.dumps(entry) + "' '" + 'https://breezytalk.firebaseio.com' + table + "'")
