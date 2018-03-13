import numpy as np
import os.path
import cv2
import pandas as pd
import time
import json
import sys
from requests_html import HTMLSession
s = HTMLSession()
import time

z = int(sys.argv[1])

if not os.path.exists("all_ships.zoom=%s" % z):
    os.mkdir("all_ships.zoom=%s" % z)

# A function to convert values from string to the most suitable format for that valuu
# - str, int or float
def mangle_type(val):
    try:
        if str(int(val)) == val.strip(): return int(val)
    except:
        pass
    try:
        if str(float(val)) == val.strip(): return float(val)
    except:
        pass
    return val

for x in range(0, 2**z):
    for y in range(0, 2**z):
        key = "%s-%s-%s" % (z, x, y)
        print(key)
        file = "all_ships.zoom=%s/%s.json" % (z, key)
        if os.path.exists(file): continue
        try:
            r = s.get(
                'https://www.marinetraffic.com/getData/get_data_json_4/z:%s/X:%s/Y:%s/station:0' % (
                    z, x, y),
                cookies={"vTo": "1"}
            )
            if r.status_code != 200:
                content = '{"status": %s}' % r.status_code
            else:
                content = r.text
        except Exception as e:
            content = '{"error": "%s"}' % e
        with open(file, "w") as f:
            f.write(content)
        time.sleep(10*np.random.random())

print("Merging data...")

data = {}
for x in range(0, 2**z):
    for y in range(0, 2**z):
        key = "%s-%s-%s" % (z, x, y)
        file = "all_ships.zoom=%s/%s.json" % (z, key)
        with open(file) as f:
            data[key] = json.load(f)

with open("all_ships.zoom=%s.json" % z, "w") as f:
    json.dump(data, f)

all_ships = pd.concat([
    pd.DataFrame(
        [{key: mangle_type(val)
          for key, val in row.items()}
         for row in item['data']['rows']])
    for item in data.values()
    if 'data' in item and 'rows' in item['data'] and item['data']['rows']
]).to_records()

np.savez_compressed("all_ships.zoom=%s.npz" % z, all_ships=all_ships)
