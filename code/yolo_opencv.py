#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import numpy as np
import pafy                # pafy lib for reading videos from the youtube


# #setup running settings
# url = 'https://www.youtube.com/watch?v=eJ6ZMd4sVrI'
# vPafy = pafy.new(url)
# play = vPafy.getbest(preftype="mp4")
# cap = cv2.VideoCapture(play.url)

# #set the dimensions of the video
# cap.set(3, 480)
# cap.set(4, 640)

cap = cv2.VideoCapture(0)


#import the labels, weights and cofiguration
classesf = "yolov3.txt"
weights = "yolov3.weights"
config = "yolov3.cfg"

scale = 0.00392
classes = None
with open (classesf, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(weights, config)


#setup the layers of output
def get_output_layers(net):         
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

#draw the prediction bounding boxes and setup it with colors and labels
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):       

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#setup the read video and detection with network
def video_detector():
   while True:
    ret, image = cap.read()
    Width = image.shape[1]
    Height = image.shape[0]
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int (detection[0] * Width)
                center_y = int (detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w/2
                y = center_y - h/2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    #reduce the boxes by NMS function
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = 0
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round (y+h))

    cv2.imshow("object detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if __name__ == "__main__":
    video_detector()

#for excution type this command below in terminal or debug
#python yolo_opencv.py --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
