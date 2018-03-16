import numpy as np
import os.path
import cv2
import pandas as pd
import time
import sys
from requests_html import HTMLSession
s = HTMLSession()

#all_ships = np.load("all_ships.zoom=3.npz")['all_ships']
all_ships = np.load(sys.argv[1])['all_ships']

# Filter out SAT-AIS ships
"""
|Fishing|ship_type_in=2
|Cargo%20Vessels|ship_type_in=7
|Tankers|ship_type_in=8
|Passenger%20Vessels|ship_type_in=6
|High%20Speed%20Craft|ship_type_in=4
|Tugs%20%26%20Special%20Craft|ship_type_in=3
|Pleasure%20Craft|ship_type_in=9
|Navigation%20Aids|ship_type_in=1
|Unspecified%20Ships|ship_type_in=0
"""
interesting_ships = all_ships[np.where(((all_ships['SHIPTYPE']==7) | (all_ships['SHIPTYPE']==8) | (all_ships['SHIPTYPE']==2)) & (all_ships['SHIP_ID']<1000000000))]
for ship in interesting_ships:
    if os.path.exists("images/%s.png" % ship['SHIP_ID']):
        continue
    try:
        print(ship['SHIPNAME'], ship['SHIP_ID'])
        cv2.imwrite(
             "images/%s.png" % ship['SHIP_ID'],
             cv2.imdecode(
                 np.frombuffer(
                     s.get("http://photos.marinetraffic.com/ais/showphoto.aspx?shipid=%s&size=" %
                       ship['SHIP_ID']).content,
                     dtype="int8"),
                 cv2.IMREAD_COLOR))
    except Exception as e:
        print(e)
    time.sleep(30*np.random.random())
