import numpy as np
import cv2
CANVAS_SIZE = (600,800)

FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)
def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("./preview/first_frame.jpg", image)
class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = cv2.imread("/home/user/Desktop/stop_line_detection/Vehicle_Detection/preview/first_frame.jpg") # Name for our window
        self.resize=cv2.resize(self.window_name,(810,640))
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.clone=self.resize.copy()
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
    def run(self):
        
        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            #canvas = np.zeros(CANVAS_SIZE, np.uint8)
            img=cv2.imread("/home/user/Desktop/stop_line_detection/Vehicle_Detection/preview/first_frame.jpg")
            resize=cv2.resize(img,(810,640))
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(resize, np.array([self.points]), False,(0,255,0), 1)
                # And  also show what the current segment would look like
                cv2.line(resize, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow("image", resize)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True
    # User finised entering the polygon points, so let's make the final drawing
        #canvas = np.zeros(CANVAS_SIZE, np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            print("points greater than zero")
            print(self.points)
            cv2.polylines(resize,np.array([self.points]),True,(0,255,),thickness=3)
           # cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        cv2.imshow("image",resize)
        # Waiting for the user to press any key
        key=cv2.waitKey(1)
        if key==ord('q'):
            print("Printing in main")
            cv2.destroyAllWindows()
        return self.points
def get_points():
    pd = PolygonDrawer("Polygon")
    pd.points = pd.run()
    
    print("Polygon points in main= %s" % pd.points)
    return pd.points
video_path='/home/user/Desktop/Red_light_violation/Red_light_violation_videos/video_4.mp4'
cap = cv2.VideoCapture(video_path)
#getFirstFrame(video_path)
points=get_points()

print(points)
net = cv2.dnn.readNetFromONNX("/home/user/Desktop/stop_line_detection/Vehicle_Detection/yolov5s.onnx")
file = open("/home/user/Desktop/stop_line_detection/Vehicle_Detection/coco.txt","r")
classes = file.read().split('\n')
#print(classes)
while True:
    img = cap.read()[1]
    img = cv2.resize(img, (810,640))
    cv2.polylines(img,np.array([points]),True,(0,255,0),5)
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
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text = label + "{:.2f}".format(conf)
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,0,255),2)
    cv2.imshow("VIDEO",img)
    k = cv2.waitKey(10)
    if k == ord('p'):
        break

