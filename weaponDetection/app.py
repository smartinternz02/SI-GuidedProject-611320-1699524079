from flask import Flask,render_template,request,session,Blueprint
import cv2
import cvzone
import math
import os 
from playsound import playsound
from ultralytics import YOLO
import time
import pygame
import threading

app = Flask(__name__)
model = YOLO("best.pt")

pygame.mixer.init()

def play_alert_sound():
    # Play a simple alert sound using pygame
    pygame.mixer.music.load('alarm.mp3')  # Replace 'alert_sound.wav' with your sound file
    pygame.mixer.music.play()
    time.sleep(5)  # Adjust duration as needed
    pygame.mixer.music.stop()

def async_play_alert_sound():
    # Run the play_alert_sound function in a separate thread
    alert_thread = threading.Thread(target=play_alert_sound)
    alert_thread.start()

@app.route('/')
def project():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/GetStarted', methods=['POST'])
def getStarted():
    return render_template('predict.html')

@app.route('/aboutUs')
def aboutUs():
    return render_template('aboutUs.html')

@app.route('/contactUs')
def contactUs():
    return render_template('contactUs.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/imagePredict')
def imagePredict():
    return render_template('imagePredict.html')

@app.route('/videoPredict')
def videoPredict():
    return render_template('videoPredict.html')

@app.route('/livePredict')
def livePredict():
    return render_template('livePredict.html')

@app.route('/videoUpload',methods=["GET","POST"])
def vidPred():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return render_template('fileCheck.html')
        basepath = os.path.dirname('__file__')
        filepath = os.path.join(basepath,"uploads",f.filename)
        print(filepath)
        f.save(filepath)
        cap =cv2.VideoCapture(filepath) 
        
        while True:
            _, frame = cap.read()
            res = model(frame, stream=True)
            for r in res:
                boxes = r.boxes
                for box in boxes:
                    #bounding box
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    w,h = x2-x1,y2-y1
                    cvzone.cornerRect(frame,(x1,y1,w,h))
                    #   confidence
                    conf = math.ceil((box.conf[0]*100))/100
                    #className
                    cls = int(box.cls[0])
                    cvzone.putTextRect(frame,f'{model.names[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)
                    if conf >= 0.7:
                        async_play_alert_sound()
            cv2.imshow("video",frame)
            if  cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return render_template('videoPredict.html')

@app.route('/imageUpload',methods=["GET","POST"])
def imgPred():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname('__file__')
        filepath = os.path.join(basepath,"uploads",f.filename)
        print(filepath)
        f.save(filepath)
        img1 = filepath
        result = model(img1,show=True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return render_template('imagePredict.html')
            
@app.route('/live', methods=['POST'])
def livePred():
        cap =cv2.VideoCapture(0) 
        while True:
            _, frame = cap.read()
            res = model(frame, stream=True)
            for r in res:
                boxes = r.boxes
                for box in boxes:
                    #bounding box
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    w,h = x2-x1,y2-y1
                    cvzone.cornerRect(frame,(x1,y1,w,h))
                    #   confidence
                    conf = math.ceil((box.conf[0]*100))/100
                    #className
                    cls = int(box.cls[0])
                    cvzone.putTextRect(frame,f'{model.names[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)
                    if conf >= 0.7:
                        async_play_alert_sound()
            cv2.imshow("video",frame)
            if  cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return render_template('livePredict.html')
if __name__ == "__main__":
    app.run(debug=True)
