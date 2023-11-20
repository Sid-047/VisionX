import os
import time
import shutil
import numpy as np
from flask import *
from deepface import DeepFace
from PIL import Image, ImageSequence

app=Flask(__name__)

@app.route('/enroll')
def EnrollPage():
    return render_template('EnrollTemplate.html')

@app.route('/enroll', methods=['POST'])
def fetch():
    global a
    if request.method=='POST':
        a=0
        gif=request.files['pic']
        print(">>>>>>>>>>>", gif.filename)
        gif.save(gif.filename)
        gif=Image.open(gif.filename)
        dir_ = "FeedData"
        try:
            shutil.rmtree(dir_)
            os.mkdir(dir_)
        except:
            os.mkdir(dir_)
        gif_frames = []
        frame_len = len(list(ImageSequence.Iterator(gif)))
        if frame_len>15:
            step = round(frame_len/15)
            print(step)
        else:
            step = 1
        print("\n\n\n****$$$$$$#####################>>>>>>>>>>>>>", frame_len)
        for frame in ImageSequence.Iterator(gif):
            gif_frames.append(np.array(frame))
        for i in gif_frames[1::step]:
            img = np.array(i)
            #print(img)
            Image.fromarray(img).save("FeedData/"+str(a)+".png")
            a+=1
        print(gif.filename)
        gif.close()
        os.remove(gif.filename)
    return redirect('/compare')

@app.route('/compare')
def ComparePage():
    return render_template('CompareTemplate.html')

@app.route('/compareModel', methods=['POST'])    
def validation():
    if request.method=='POST':
        pic=request.files['pic']
        pic.save(pic.filename)
        acc=np.array([])
        dir_ = os.getcwd()+"\\FeedData"
        for i in os.walk(dir_):
            for j in i[2]:
                print("................>>>>>>>>>>>>",j)
                try:
                    res = DeepFace.verify(img1_path=dir_+"\\"+j, img2_path=os.getcwd()+"\\"+pic.filenae, model_name="Facenet512", detector_backend="ssd", distance_metric="euclidean_l2")
                except:
                    continue
                print(res)
                acc = np.append(acc,np.array(res["verified"]))
        os.remove(os.getcwd()+"\\"+pic.filename)
        accuracy = sum(list(acc))/len(list(acc))
        if accuracy>0.6:
            result = True
        else:
            result = False
        return str(accuracy)+" --- "+str(result)
        
if __name__=='__main__':
    app.run(debug=True)
    
