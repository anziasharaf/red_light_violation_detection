import numpy as np
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import timedelta
import datetime
CANVAS_SIZE = (600,800)
no_of_polygons=2

FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = "/home/user/Desktop/projects/stop_line_detection/Vehicle_Detection/preview/reference_frame_107.png" # Name for our window
        #self.window_name=cv2.resize(self.window_name,(810,640))
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.polygons=[] # List of polygon coordinates
        self.no_of_poygons=1 #number of polygons
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.on_mouse)
        

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True
            #self.polygons.append(self.points)
            #print("polygon cordinates",self.polygons)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.done=False
            self.polygons.append(self.points)
            self.points.clear()
            #  double click left button adds new polygon
            self.polygon = self.polygon + 1
            self.points.append((x, y))
    def run(self):

        #cv2.namedWindow('image')
        #cv2.setMouseCallback('image', self.on_mouse)
        
        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            #canvas = np.zeros(CANVAS_SIZE, np.uint8)
             # Name for our window
            img=cv2.imread("/home/user/Desktop/projects/stop_line_detection/Vehicle_Detection/preview/reference_frame_107.png")
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(img, np.array([self.points]), False,(0,255,0), 1)
                # And  also show what the current segment would look like
                cv2.line(img, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow("image", img)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True
    # User finised entering the polygon points, so let's make the final drawing
        #canvas = np.zeros(CANVAS_SIZE, np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            print("points greater than zero")
            print("points",self.points)
            cv2.polylines(img,np.array([self.points]),True,(0,255,0),thickness=3)
           # cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        #coordinates=[self.points[0],self.points[1],self.points[2],self.points[3],self.points[4],self.points[5]]
        print("cordinates",self.points)
        cv2.imshow("image",img)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow("image")
        return self.window_name,self.points
#if __name__ == "__main__":
def get_points():
    pd = PolygonDrawer("Polygon")
    image,cordinates = pd.run()
    print("cordinates in main",cordinates)
    return cordinates
video_path='rtsp://admin:123456@192.168.2.107:554'
cap = cv2.VideoCapture(video_path)
points=get_points()
print("cordinates in main points",points)
cord_1=[points[0],points[1],points[2],points[3]]
polygon1 = Polygon(cord_1)
print(cord_1)
#cord_2=[(479,241),(337,297),(818,718),(1040,632)]
#polygon2=Polygon(cord_2)
#print(cord_2)
net = cv2.dnn.readNetFromONNX("/home/user/Desktop/projects/stop_line_detection/Vehicle_Detection/yolov5s.onnx")
file = open("/home/user/Desktop/projects/stop_line_detection/Vehicle_Detection/coco.txt","r")
classes = file.read().split('\n')
while True:
    img = cap.read()[1]
    #img = cv2.resize(img, (810,640))
    cv2.polylines(img,np.array([cord_1]),True,(0,255,0),3)
    #cv2.polylines(img,np.array([cord_2]),True,(255,0,0),3)
    #cv2.polylines(img,np.array([cord_3]),True,(255,0,0),3)

    #cv2.polylines(img,np.array([cord_3]),True,(0,0,255),3)
    #print("points",points)
    if img is None:
            break
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]
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
            print("cordinates",x1,y1,w,h)
            c_x=x1+w/2
            c_y=y1+h/2
            label = classes[classes_ids[i]]
            pt=Point(c_x,c_y)
            check_point=pt.within(polygon1)
            print(check_point)
            if check_point==True:
                conf = confidences[i]
                text = label + "{:.2f}".format(conf)
                cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,0,255),2)
                print(x1,y1,w,h)
                millisecs=cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = timedelta(milliseconds=millisecs)
                
                print(timestamp)
                dt = str(datetime.datetime.now())
                print(dt)
                print('violation at millisecs',str(dt))
                filename='frame_'+str(millisecs)+'.jpg'
                cv2.imwrite('./captured_frame/'+filename,img)
            #clip_video(int(millisecs))
            #print(timestamp)
    cv2.imshow("VIDEO",img)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break