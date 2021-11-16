
import datetime
from flask import Flask , render_template ,request   
import cv2 as cv
from testimages import *  # self defined library to test faces in uploaded images
from train import *      # self defined library to train new faces
from PIL import Image
import os
import csv

# file location of record file
file = 'static//records.csv'



app = Flask(__name__)

@app.route("/", methods=['POST','GET'])    #<-- home page
def hello():
    img_list = os.listdir('static/train/')
    if request.method=='POST':
        img = request.files.get('imagefile', '')
        name = request.form['username']
        img = Image.open(img.stream)
        img_loc = f'static//temp//{name}.jpg'
        img.save(img_loc)    
        img_list = os.listdir('static/train/')
        status =train(name)  # train the model with new image
        if status == "nf":
            return render_template('/new.html' ,newmsg=0 , msg = "No Face Found in Uploaded Image Pls ReUpload")
        if status == "al":
            return render_template('/new.html' ,newmsg=0 , msg = "Face Already registerd")
        return render_template('/new.html'  ,newmsg=1)
    return render_template('/index.html' , images= img_list)


@app.route("/upload", methods=['POST','GET'])     # page to upload image for test faces with available trained model
def upload():
    if request.method=='POST': 
        img = request.files.get('imagefile', '')
        img = Image.open(img.stream)
        img_loc = 'static//saved.jpg'
        img.save(img_loc)
        testimg = face_recognition.load_image_file(img_loc)
        img = cv.cvtColor(testimg , cv.COLOR_BGR2RGB)
        img , (count , t_count) ,fname  ,confidance , path= test_img(img)  
        print("Total Faces in Img :" , t_count)
        print("Total Faces Matched in Img :" , count)
        if t_count != 0:
            date = str(datetime.datetime.now())
            f = open("static//records.csv", "a")
            for name in fname:
                 f.write(f"{name},{date[0:10]},{date[11:19]}\n")
            f.close()
        
   
    return render_template('/out.html' , img = 'saved.jpg' , face = fname , total = count ,t_count =t_count, imgpath = path , confidance = confidance)


@app.route("/newtrain", methods=['POST','GET'])    # page to train model with new image
def newt():
    return render_template('/new.html' )


@app.route("/record", methods=['POST','GET'])     # page to view record of all users
def rec():
    val = []
    with open("static//records.csv", "r") as f:
        reader = csv.reader(f)
        for i in reader:
            val.append(i)
    return render_template('/records.html' , val = val)


# app.run(debug=True , host ='0.0.0.0' , port = "8080")  
app.run(debug=True )  


cv.waitKey(0) 


