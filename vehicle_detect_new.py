import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
#from draw_rect import main
#Web Camera
f=open("detect_list.csv","w")
cap = cv2.VideoCapture('/home/user/Desktop/Red_light_violation/Vehicle-detection/assets/v1.mp4')

min_width_rectangle = 80
min_height_rectangle = 80

count_line_position = 550
# Initialize Substructor
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy
count=0
my_dict={}
result=[]
detect = []
offset = 6 #Alowable error b/w pixel
counter = 0
model =load_model('new_model.h5')
while True:
    ret, video = cap.read()
    print(ret)
    if not ret:
        print(ret)
        #print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 5)
    
# Applying on each frame
    vid_sub = algo.apply(blur)
    dilat = cv2.dilate(vid_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    countersahpe, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(video, (25,count_line_position),(1200,count_line_position),(255,0,0), 3)
    #main()
    count=0
    for (i, c) in enumerate(countersahpe):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=min_width_rectangle) and (h>= min_height_rectangle)
        if not val_counter:
            continue
        crop_img=video[y:y+h,x:x+w]
        img = cv2.resize(crop_img,(224,224))
        img = img.reshape(224,224,3)
        img_pred = np.expand_dims(img, axis = 0)
        rslt = model.predict(img_pred)
        print(rslt)
        
        cv2.imwrite('./objects/image%d.jpg' %count,crop_img)
        file_name='image{}.jpg'.format(count)
        #f.write(file_name+'\t'+rslt+'\n')
        count+=1
        my_dict['f_name']=file_name
        my_dict['value']=rslt
        result.append(my_dict)
        #f.write('image{}.jpg'.format(count)+'\t'+rslt+'\n')
        if rslt[0][0] == 1:
            print("vehicle")
            cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,255),2)
            #cv2.putText(video,"Vehicle No: " + str(counter), (x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)
            center = center_handle(x,y,w,h)
            detect.append(center)
            cv2.circle(video, center, 4, (0,0,255), -1)
            for (x,y) in detect:
                if y<(count_line_position + offset) and  y>(count_line_position - offset):
                   counter+=1
                   #cv2.line(video, (25,count_line_position),(1200,count_line_position),(0,127,255), 3)
                   detect.remove((x,y))
                   print("Vehicle No: "+ str(counter))
                   #cv2.putText(video,"Vehicle No: " + str(counter), (450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    cv2.imshow('Detector',video)

    #print(result)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
#https://mlhive.com/2022/04/draw-on-images-using-mouse-in-opencv-python
