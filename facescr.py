#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facescr.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2,os
import time
# local modules
# from video import create_capture
from common import clock, draw_str


from mss import mss
from PIL import Image
import io

# mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}
# sct = mss()
# while 1:
#     sct.get_pixels(mon)
#     img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
#     cv2.imshow('test', np.array(img))
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

dbfilepath = "scrshut/trainner.yml"

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    # print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    # try:
    #     video_src = video_src[0]
    # except:
    #     video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    mon = {'top': 100, 'left': 100, 'width': 200, 'height': 200}
    sct = mss()


    recognizer = cv2.face.createLBPHFaceRecognizer()
    if os.path.exists(dbfilepath):
        recognizer.load(dbfilepath)
    else:
        pilImage=Image.open("scrshut/rootimg.jpg").convert('L')
        imageNp=np.array(pilImage,'uint8')
        recognizer.train( [imageNp], np.array([1]) ) 
        recognizer.save(dbfilepath)

    

#    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    while True:
#        ret, img = cam.read()
        sct.get_pixels(mon) #screen rect capturing
        img = np.array( Image.frombytes('RGB', (sct.width, sct.height), sct.image) )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        t = clock()
        rects = detect(gray, cascade)
        # vis = img.copy()
        draw_rects(img, rects, (0, 255, 0))

        for x, y, w, h in rects:
            roi = gray[y:h, x:w]
            # vis_roi = vis[y1:h, x1:w]
            # subrects = detect(roi.copy(), cascade)
            # draw_rects(vis_roi, subrects, (255, 0, 0))

            cv2.imshow('facedetect2', roi)
            Id, conf = recognizer.predict(roi)
            if(conf<50):
                if(Id==1):
                    Id="Andrii"
                elif(Id==2):
                    Id="Andrey"
                elif(Id==4):
                    Id="Yeva"
                elif(Id==1496186763):
                    Id="D.Serega"
                elif(Id==1496186730):
                    Id="M.Lena"
            elif(conf > 50 and conf<100): #undefined        
                recognizer.update( np.array([roi]), np.array([Id]) ) 
                recognizer.save(dbfilepath)
            elif(conf > 100): #undefined
                Id="NEW"
                ts = int(time.time())
                recognizer.update( np.array([roi]), np.array([ts]) ) 
                recognizer.save(dbfilepath)
            # else:
            #     Id="??"        
            cv2.putText(img, str(Id) ,(x-20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2, cv2.LINE_AA)
            cv2.putText(img, "%.1f" % conf ,(x-20, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(200,0,0),2, cv2.LINE_AA)


        # for(x,y,w,h) in faces:
        #     cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        #     Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        #     if(conf<50):
        #         if(Id==1):
        #             Id="Andrii"
        #         elif(Id==2):
        #             Id="Andrey"
        #         elif(Id==4):
        #             Id="Yeva"
        #     else:
        #         Id="??"        
        #     cv2.putText(im, str(Id),(x, y+h), font, 4,(255,255,255),2, cv2.LINE_AA)

        # // eyes rect
        # if not nested.empty():
        #     for x1, y1, x2, y2 in rects:
        #         roi = gray[y1:y2, x1:x2]
        #         vis_roi = vis[y1:y2, x1:x2]
        #         subrects = detect(roi.copy(), nested)
        #         draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(img, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', img)

        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
    recognizer.save(dbfilepath)
