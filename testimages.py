import time
from PIL import Image
import cv2 as cv
import face_recognition
import pickle
from array import *

def test_img (img):
    img = cv.resize(img, (800,800))
    facename= []
    with open('faceDictionary.pickle', 'rb') as f:
        facedic =pickle.load(f)
    available_faces = len(facedic)
    t= time.time()
    new_img_path = f'static/upload/{t%1}.jpg'
    total_face_count = 0
    confidance= []
    faceloctest = face_recognition.face_locations(img)   # returns top , right ,bottom , left
    print( " total faces found : ", len(faceloctest) )
    
    if len(faceloctest) == 0:
        cv.putText(img , "No Face Found " , (100,100) , 0 , 2 , (255,0,0) ,5)
        cv.imwrite(new_img_path, img)
        print("no training images availabel")
        return img , (0 ,0) ,"No Face Found" , " 0" , new_img_path
    for i in range(len(faceloctest)):
        print(f"For loop for {i} Face ")
        cv.rectangle(img, (faceloctest[i][3],faceloctest[i][0]) , (faceloctest[i][1],faceloctest[i][2]) , (0,255,0) , 2)
        try:
            faceencodetest = face_recognition.face_encodings(img)[i]
        except Exception as e:
            print("Error Occured as ",e )
            continue
            
        # print( "test encode",faceencodetest)
        key = list(facedic.keys())
        compared  = False
        for k in key:
            train_encode = facedic[k]
            compare = face_recognition.compare_faces([train_encode] , faceencodetest)
            # print(f"comparing face with {k} and get =", compare)
            if compare==[True] :
                confidance.append(int(face_recognition.face_distance([train_encode] , faceencodetest)[0]*100))
                print(" face matched with ",k )
                compared  = True
                facename.append(k)
                cv.rectangle(img, (faceloctest[i][3],faceloctest[i][0]) , (faceloctest[i][1],faceloctest[i][2]) , (0,0,255) , 2)
                cv.putText(img , f"face :{k} " , (faceloctest[i][3],faceloctest[i][0] -3 ) , 0 , 0.5 , (0,0,0) ,2)
                cv.putText(img , f" with {int(face_recognition.face_distance([train_encode] , faceencodetest)[0]*100)}% " , (faceloctest[i][3],faceloctest[i][0]+13) , 0 , 0.5 , (0,0,0) ,1)
                total_face_count =total_face_count+1
        if compared==False:
            cv.putText(img , "Unidentified" , (faceloctest[i][3],faceloctest[i][0] -3 ) , 0 , 0.5 , (255,0,0) ,2)
    
    cv.imwrite(f"{new_img_path}" , img)
    print("totalconf************* : ", confidance)
    return img , (total_face_count ,len(faceloctest)) ,facename ,confidance ,new_img_path



        
