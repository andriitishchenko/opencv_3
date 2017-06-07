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
import sqlite3

# http://answers.opencv.org/question/19763/storing-opencv-image-in-sqlite3-with-python/
# https://stackoverflow.com/questions/13211745/detect-face-then-autocrop-pictures

dbSQLpath =      "scrshut/db.sqlite"
untitledpath =   "scrshut/sugest/"
rec_db_unknown = "scrshut/unknown.yml"
rec_db_known =   "scrshut/known.yml"

def get_profile(labelID):
    sql="select * from records where labelID="+str(labelID)
    rez=sqlDBConn.execute(sql)
    data=None
    for item in rez:
        data=item
    return data

def add_profile(labelID,labelTitle):
    if get_profile(labelID)==None:
        sqlDBConn.execute('insert into records values (?,?)', [str(labelID),str(labelTitle)] )
        sqlDBConn.commit()
    return
    
def detect(img, cascade): #1.3, 4
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def train_update(id):
    path="%s%s/_*"%(untitledpath,str(id)) 
    procesed=0 
    for file in glob.glob(path):
        pilImage=Image.open(file)
        imageNp=np.array(pilImage,'uint8')
        rec_unknown.update( [imageNp], np.array([id]) ) 
        os.remove(file)
        procesed=1
    if procesed==1:
        shutil.rmtree("%s%s/"%(untitledpath,str(id)))
        # rec_unknown.save(rec_db_unknown)
    return        

def init_rec_db(path):
    rec = cv2.face.createLBPHFaceRecognizer()
    if os.path.exists(path):
        rec.load(path)
    else:
        pilImage=Image.open("scrshut/train.jpg").convert('L')
        pilImage_np=np.array(pilImage,'uint8')
        # gray = cv2.cvtColor(pilImage_np, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(pilImage_np)
        rects = detect(gray, cascade)
        faces=[]
        lbls=[]
        for x, y, w, h in rects:
            roi = gray[y:h, x:w]
            faces.append(roi)
            lbls.append(1)
            # rec.update( np.array([roi]) , np.array([1]) ) 
            # cv2.imshow('facedetect='+str(x), roi)
        rec.train(faces, np.array(lbls))
        rec.save(path)
        add_profile(1,"face")
    return  rec


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

    sqlDBConn = sqlite3.connect(dbSQLpath)

    mon = {'top': 100, 'left': 100, 'width': 400, 'height': 300}
    sct = mss()

    rec_unknown = init_rec_db(rec_db_unknown)
    rec_known = init_rec_db(rec_db_known)


    # rec_unknown = cv2.face.createLBPHFaceRecognizer()
    # if os.path.exists(rec_db_unknown):
    #     rec_unknown.load(rec_db_unknown)
    # else:
    #     pilImage=Image.open("scrshut/rootimg.jpg").convert('L')
    #     imageNp=np.array(pilImage,'uint8')
    #     rec_unknown.train( [imageNp], np.array([1]) ) 
    #     rec_unknown.save(rec_db_unknown)
    #     add_profile(1,"face")

    # rec_known = cv2.face.createLBPHFaceRecognizer()
    # if os.path.exists(rec_db_known):
    #     rec_unknown.load(rec_db_known)
    # else:
    #     pilImage=Image.open("scrshut/rootimg.jpg").convert('L')
    #     imageNp=np.array(pilImage,'uint8')
    #     rec_unknown.train( [imageNp], np.array([1]) ) 
    #     rec_unknown.save(rec_db_known)
    #     add_profile(1,"face")

    

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

            user = None

            # cv2.imshow('face', roi) #active face !!FOR ONE FACE PER PIC
            true_id, conf = rec_known.predict(roi)

            if conf < 50: #known face
                user = get_profile(true_id)
            else:   #search in sugestions
                Id, conf = rec_unknown.predict(roi)
                user = get_profile(Id)
                if conf < 50:
                    rec_known.update( np.array([roi]), np.array([Id])) 
                    rec_known.save(rec_db_known)
                elif(conf > 100): #undefined
                    isnewface=raw_input('new face? y|n: ')
                    if isnewface == "y":
                        ts = int(time.time())
                        rec_unknown.update( np.array([roi]), np.array([ts]) ) 
                        rec_unknown.save(rec_db_unknown)
                        add_profile(ts,ts)
                else : # request for train
                    isnupdate=raw_input( "is this '%s' %.1f ?  y|n:"%(('None' if user==None else str(user[1]) ) ,conf) )
                    if isnupdate == "y":
                        rec_unknown.update( np.array([roi]), np.array([Id]) ) 
                        rec_unknown.save(rec_db_unknown)

            
                    
            # if(conf > 100): #undefined face
            #     isnewface=raw_input('new face? y|n: ')
            #     if isnewface == "y":
            #         ts = int(time.time())
            #         rec_unknown.update( np.array([roi]), np.array([ts]) ) 
            #         rec_unknown.save(rec_db_unknown)
            #         add_profile(ts,ts)
            # elif(conf > 70 and conf<90): #shugested, save for aprove
            #     directory = untitledpath+str(Id)
            #     fname=directory+"/%.0f.jpg" % conf
            #     if not os.path.exists(fname):
            #         if not os.path.exists(directory):
            #             os.makedirs(directory)
            #         face=Image.fromarray(roi)
            #         face.save(fname)
            #     train_update(Id)

            # if(conf<50):

            # elif(conf > 50 and conf<100): #undefined        
            #     rec_unknown.update( np.array([roi]), np.array([Id]) ) 
            #     rec_unknown.save(rec_db_unknown)
            # elif(conf > 100): #undefined
            #     Id="NEW"
            #     ts = int(time.time())
            #     rec_unknown.update( np.array([roi]), np.array([ts]) ) 
            #     rec_unknown.save(rec_db_unknown)
            # else:
            #     Id="??"  

            cv2.putText(img, 'None' if user==None else str(user[1]) ,(x-20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2, cv2.LINE_AA)
            cv2.putText(img, "%.1f" % conf ,(x-20, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,200,0),2, cv2.LINE_AA)


        # for(x,y,w,h) in faces:
        #     cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        #     Id, conf = rec_unknown.predict(gray[y:y+h,x:x+w])
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
# exit
    cv2.destroyAllWindows()
    rec_unknown.save(rec_db_unknown)
    rec_known.save(rec_db_known)
    sqlDBConn.close()

