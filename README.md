# opencv_3
Demo of face recognition with opencv 3 + python


preparation

> brew tap homebrew/science
> sudo easy_install pip
> [sudo]pip install --upgrade pip
> [sudo]pip install image
> [sudo]pip install pillow



step 1 

>mkdir dataSet

>mkdir trainner

>python 1_dataSet.py

Input integer value.

This will generate 20 images with webcam for training recognition:

dataSet/*.jpg

step 2
>python 2_trainer.py
will create a recognition yml config 
trainner/trainner.yml

step 3
>python 3_detect.py
try to recognize
Modify the ids in script

all benefits to this guy:
https://www.youtube.com/watch?v=1Jz24sVsLE4&list=PLnjEM1fs09cGGjdCLSue8Kw7GmWDhGlMh