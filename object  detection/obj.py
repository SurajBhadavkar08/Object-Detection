#Import opencv and numpy library
import cv2
import numpy as np
import datetime
import matplotlib
import argparse
import os
from gtts import gTTS
from playsound import playsound
import pygame
pygame.mixer.init()
score = 0
#import tensorflow as tf
#print (tf.version.VERSION)
#Import yolov4 model
net = cv2.dnn.readNetFromDarknet('C:\Python\yolov4.cfg', 'C:\Python\yolov4.weights')
ln = net.getLayerNames()
ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]
DEFAULT_CONFIANCE = 0.5
THRESHOLD = 0.4
#import object names file
#with open('C:\Python\labels.txt', 'r') as f:
    #LABELS = f.read().splitlines()
classes = []
with open("C:\Python\labels.txt", 'r') as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
    #print(LABELS)
#Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1928)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1888)
button_person = False
def click_button(event,x,y,flags,params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
       print(x,y)
       polygon=np.array([(20, 20), (220, 20), (220, 70), (20, 70)])
       is_inside = cv2.pointPolygonTest(polygon, (x, y), False)
       if is_inside > 0:
           print("you are clicking in the button")
           if button_person is False:
               button_person = True
           else:
               button_person = False
           print("Now button person is: ", button_person)
# Create windows
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", click_button)


#Running loop until user press (q,Q) key
while True:
    _,image=cap.read()
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > DEFAULT_CONFIANCE:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, W, H) = box.astype("int")
                x = int(centerX - (W / 2))
                y = int(centerY - (H / 2))
                boxes.append([x, y, int(W), int(H)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIANCE, THRESHOLD)
    COLORS = np.random.uniform(0,255,size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            (x, y, w, h) = boxes[i]
            color = COLORS[i]
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            datet = str(datetime.datetime.now())

            #Draws rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            #Put text/object name on top of the rectangle.
            cv2.putText(image, text, (x, y - 10 ), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            cv2.putText(image,datet, (370,20), cv2.FONT_HERSHEY_PLAIN,1, color,2)
        if button_person is True and not pygame.mixer.music.get_busy() and class_name == class_name and score>.5:
            score +=1
            name = class_name +".mp3"
            #search for classname using google if needed
            if not os.path.isfile(name):
                tts = gTTS(text="I see a " + class_name, lang='en', slow=True)
                tts.save(name)
            last = 0
            pygame.mixer.music.load(name)
            pygame.mixer.music.play()
        #create button
        cv2.rectangle(image, (20, 10), (400, 80), (255, 0, 255), -1)
        polygon = np.array([(20, 20), (220, 20), (220, 70), (20, 70)])
        cv2.putText(image,"Detect objects", (30, 60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255,255,255), 3 )
        polygon = np.array([(20, 20), (220, 20), (220, 70), (20, 70)])
        cv2.putText(image, "isense", (500, 1005), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 4, (255,255,255), 2 )


#Shows screen on display
    cv2.imshow('Image', image)
    if cv2.waitKey(20)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
