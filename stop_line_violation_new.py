import cv2
import numpy as np
from datetime import timedelta
import datetime
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
def clip_video(ti):
   
   ffmpeg_extract_subclip("/home/user/Desktop/projects/stop_line_detection/Red_light_violation_videos/video_2.mp4",int(ti)-5, int(ti), targetname="test.mp4")
offset=75
violation=False
def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("./preview/first_frame.jpg", image)
class DrawLineWidget(object):
    def __init__(self):
        self.original_image = cv2.imread('./preview/first_frame.jpg')
        #self.clone = self.original_image.copy()
        self.resize=cv2.resize(self.original_image,(810,640))
        self.clone=self.resize.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []
    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone) 

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()
    def show_image(self):
        return self.clone
def get_points():
    draw_line_widget = DrawLineWidget()
    while True:
       cv2.imshow('image', draw_line_widget.show_image())
       key = cv2.waitKey(1)
       # Close program with keyboard 'q'
       if key == ord('q'):
          print("Printing in main")
          cor=(draw_line_widget.image_coordinates[0],draw_line_widget.image_coordinates[1])
          #print(draw_line_widget.image_coordinates[0],draw_line_widget.image_coordinates[1])
          cv2.destroyAllWindows()
          break
    return  cor
def stop_line_detection():
    video_path='/home/user/Desktop/projects/stop_line_detection/Red_light_violation_videos/video_1.mp4'
    cap = cv2.VideoCapture(video_path)
    getFirstFrame(video_path)
    points=get_points()
    print(points)
    print(points[0],points[1])
    net = cv2.dnn.readNetFromONNX("/home/user/Desktop/projects/stop_line_detection/Vehicle_Detection/yolov5s.onnx")
    file = open("/home/user/Desktop/projects/stop_line_detection/Vehicle_Detection/coco.txt","r")
    classes = file.read().split('\n')
    #print(classes)

    while True:
        img = cap.read()[1]
        img = cv2.resize(img, (810,640))
        cv2.line(img,points[0],points[1],(0,255,0),3)
        #print("point1",points[0][1])
        #print("point2",points[1][1])
        y_cord1=points[0][1]
        y_cord2=points[1][1]
        x_cord1=points[0][0]
        x_cord2=points[0][1]
        diff=y_cord1-y_cord2
        if img is None:
            break
        blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
        net.setInput(blob)
        detections = net.forward()[0]
    
        # cx,cy , w,h, confidence, 80 class_scores
        # class_ids, confidences, boxes

        classes_ids = []
        confidences = []
        boxes = []
        rows = detections.shape[0]

        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width/640
        y_scale = img_height/640

        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.5:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.5:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx- w/2)*x_scale)
                    y1 = int((cy-h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1,y1,width,height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)
        interested_class_ids=[1,2,3,5,6,7]
        for i in indices:
            if classes_ids[i] in interested_class_ids:
                x1,y1,w,h = boxes[i]
                label = classes[classes_ids[i]]
                #print(y_cord1,y_cord1-offset,y1)
                value=(x_cord2-x_cord1)*(y1-y_cord1)-(x1-x_cord1)*(y_cord2-y_cord1)
                print(value)
                print(x_cord2)
                if x1>x_cord1 :
                    if y1>y_cord1-150:
                        if value<0:
                            conf = confidences[i]
                            text = label + "{:.2f}".format(conf)
                            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,0,255),2)
                            print(x1,y1,w,h)
                            millisecs=cap.get(cv2.CAP_PROP_POS_MSEC)
                            timestamp = timedelta(milliseconds=millisecs)
                            dt = str(datetime.datetime.now())
                            print(dt)
                            filename='frame_'+str(millisecs)+'.jpg'
                            cv2.imwrite('./captured_frame/'+filename,img)
                            clip_video(int(millisecs))
                            print(timestamp)
                            print("Violation Detected at: "+str(dt))
                
        cv2.imshow("VIDEO",img)
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
stop_line_detection()
