# opencv_3
Demo of face recognition with opencv 3 + python


preparation

> brew tap homebrew/science<br />
> sudo easy_install pip<br />
> [sudo]pip install --upgrade pip<br />
> [sudo]pip install image<br />
> [sudo]pip install pillow<br />



step 1 

> mkdir dataSet<br />
> mkdir trainner<br />
> python 1_dataSet.py<br />

Input integer value.<br />
This will generate 20 images with webcam for training recognition:<br />

dataSet/*.jpg<br />

step 2

> python 2_trainer.py<br />

will create a recognition yml config <br />
trainner/trainner.yml<br />

step 3

>python 3_detect.py<br />

try to recognize<br />
Modify the ids in script<br />

all benefits to this guy:<br />
https://www.youtube.com/watch?v=1Jz24sVsLE4&list=PLnjEM1fs09cGGjdCLSue8Kw7GmWDhGlMh
