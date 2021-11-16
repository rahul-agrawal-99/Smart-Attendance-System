import cv2 as cv
import face_recognition
import os
import pickle
from array import *
import shutil
import dlib

detector = dlib.get_frontal_face_detector()

print("*************    Training Data  ****************************")
def train(img_name):
    loc ="static//temp//"+ img_name + ".jpg"
    testimg = face_recognition.load_image_file(loc)
    img = cv.cvtColor(testimg , cv.COLOR_BGR2RGB)
    faceloctest = detector(img)
    if len(faceloctest) == 0:
        print("No face found in the image")
        return "nf"
    with open('faceDictionary.pickle', 'rb') as f:
        faceDictionary = pickle.load(f)
    available_encodings = faceDictionary.keys()
    # print("Available Encodings: ", available_encodings)
  
    if img_name in available_encodings:
            print("Already trained")
            return "al"
    img = "static//temp//"+ img_name + ".jpg"
    print("Training model on Image: ", img)
    myimg = face_recognition.load_image_file(img)
    img = cv.cvtColor(myimg , cv.COLOR_BGR2RGB)
    #  get face location 
    faceloc = face_recognition.face_locations(img)[0]   # returns top , right ,bottom , left
    # to draw rectangle on face
    # cv.rectangle(img , (faceloc[3],faceloc[0]) , (faceloc[1],faceloc[2]) , (0,255,0) , 2)
    #  get face encodings
    faceencode = face_recognition.face_encodings(myimg)[0]                
    faceDictionary[img_name]= faceencode.tolist()
    with open('faceDictionary.pickle', 'wb') as f:
        pickle.dump(faceDictionary, f, pickle.HIGHEST_PROTOCOL)
    shutil.move("static//temp//"+ img_name + ".jpg" , "static//train//"+ img_name + ".jpg")
    # os.remove("static//temp//"+ img_name + ".jpg")
    return True

train("ree")