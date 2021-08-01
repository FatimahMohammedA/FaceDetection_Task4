# FaceDetection_Task4

Real-time-Face-Detection-by-OpenCV

This project is done using Pycharm IDE and Python, it is a Real-Time Face recognition using OpenCV while performing object detection using Haar feature-based cascade classifiers for detecting face, eyes, and smile.

1. Import and initialize

Start by importing OpenCV and create a directory (ex: Cascades) to gather all Haar classifiers files that you want to use in you project, then use their path to load them into your project.

Note: you can find Haar cascade classifiers files here (https://github.com/opencv/opencv/tree/master/data/haarcascades)

import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_FRONTALFACE_DEFAULT.xml')

eye_cascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_EYE.xml')

smileCascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_SMILE.xml')


2. Setting up your camera

To start we need to capture the face and to do so we are using the PC embedded camera which we are referring to it using (0) & setting the window size to specific measures in the following code lines:

cap = cv2.VideoCapture(0)

cap.set(3,640) # set Width

cap.set(4,480) # set Height


3. Call the classifier function


We will set our camera and inside the loop, load our input video in grayscale mode then we must call our classifier function, passing it some very important parameters, as scale factor, number of neighbors and minimum size of the detected face.

while True:

    ret, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
    
        gray,
        
        scaleFactor=1.2,
        
        minNeighbors=5,
        
        minSize=(20, 20)
    )
    
    
4. Detecting Faces

The function will detect faces on the image. Next, we must "mark" the faces in the image, using, for example, a blue rectangle. If faces are found, it returns the positions of detected faces as a rectangle with the left up corner (x,y) and having "w" as its Width and "h" as its Height ==> (x,y,w,h). This is done with this portion of the code:

for (x,y,w,h) in faces:

    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    roi_gray = gray[y:y+h, x:x+w]
    
    roi_color = img[y:y+h, x:x+w]
    
    
    
    <img width="870" alt="FaceDetection" src="https://user-images.githubusercontent.com/86549177/127755866-aa84ffde-45b0-4e79-a843-96235c8320d9.png">
    
    
