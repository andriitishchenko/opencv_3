#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facescr.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function
import math
import numpy as np
import cv2,os
import time
import glob 
# local modules
# from video import create_capture
from common import clock, draw_str


from mss import mss
from PIL import Image
import io
import shutil

untitledpath = "scrshut/durty/"
    
def detect(img, cascade): #1.3, 4
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def save(img, name):
    Id="test/"
    directory = untitledpath+str(Id)
    fname=directory+"/%s.jpg" % name
    if not os.path.exists(fname):
        if not os.path.exists(directory):
            os.makedirs(directory)
        face=Image.fromarray(img)
        face.save(fname)
    return
  
if __name__ == '__main__':
    import sys, getopt
    # print(__doc__)

    args, _src = getopt.getopt(sys.argv[1:], '', ['cascade='])
    try:
        _src = _src[0]
    except:
        _src = "/Users/andrux/Documents/Eva_1_September"
        # sys.exit(1)

    print(_src)

    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascade_frontalface_alt.xml")
    cascade = cv2.CascadeClassifier(cascade_fn)
    incr =0 
    path="%s/*.jpg"%_src 
    for file in glob.glob(path):
        print(file)        
        pilImage=Image.open(file).convert('L')
        imageNp=np.array(pilImage,'uint8')
        gray = cv2.equalizeHist(imageNp)
        rects = detect(gray, cascade)
        for x, y, w, h in rects:
            roi = gray[y:h, x:w]         
            save(roi,incr)
            incr+=1

