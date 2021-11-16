from PIL import Image
import cv2 as cv
import face_recognition
import pickle
from array import *

def test_img (img):

    facename= []
    with open('faceDictionary.pickle', 'rb') as f:
        facedic =pickle.load(f)
    available_faces = len(facedic)
    if available_faces == 0 :
        print("no training images availabel")
        return 
    #  get face location 
    total_face_count = 0
    confidance= []
    faceloctest = face_recognition.face_locations(img)   # returns top , right ,bottom , left
    print( " total faces found : ", len(faceloctest) ,end=" ")
    for i in range(len(faceloctest)):
        cv.rectangle(img, (faceloctest[i][3],faceloctest[i][0]) , (faceloctest[i][1],faceloctest[i][2]) , (0,255,0) , 2)
    #  get face encodings
        faceencodetest = face_recognition.face_encodings(img)[i]
        # print( "test encode",faceencodetest)
        key = list(facedic.keys())
        for k in key:
            train_encode = facedic[k]
            compare = face_recognition.compare_faces([train_encode] , faceencodetest)
            
            confidance.append(int(face_recognition.face_distance([train_encode] , faceencodetest)[0]*100))
        # print(compare)
            if compare==[True] :
                print(" face matched with ",k )
                facename.append(k)
                # cv.rectangle(img, (faceloctest[i][3],faceloctest[i][0]) , (faceloctest[i][1],faceloctest[i][2]) , (0,0,255) , 2)
                cv.putText(img , f"face :{k} " , (faceloctest[i][3],faceloctest[i][0] -3 ) , 0 , 0.5 , (255,255,255) ,1)
                cv.putText(img , f" with {confidance}% " , (faceloctest[i][3],faceloctest[i][0]+13) , 0 , 0.5 , (255,255,255) ,1)
                total_face_count =total_face_count+1

    return img , (total_face_count ,len(faceloctest)) ,facename ,confidance



        
