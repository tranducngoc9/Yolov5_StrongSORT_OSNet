# importing libraries
import cv2
import numpy as np
import keyboard


# Read until video is completed
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # x1 += offset[0]
        # y1 += offset[1]
        # x2 += offset[0]
        # y2 += offset[1]
        
        id = int(identities) if identities is not None else 0
        
        color = compute_color_for_labels(id)
        
        label = f'{id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        #draw bounding box
        cv2.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), color, 3) 
        #draw box label
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img
#956
# read text
data = []
with open ("runs/track/exp/tracks/2022-10-05-09-30_cut.txt", "r") as f:
    f = f.read()
    f = f.split("\n")

for i in f:
    i = i.split(" ")
    data.append(i)


i = 790
while(True):

    # Capture frame-by-frame
    
    i +=1
    frame = cv2.imread("list_img/"+ str(i) + ".jpg")
    print(i)
    for boxes in data:
        if boxes[0] == str(i):
            print(boxes)
            # Draw boxq
            frame = draw_boxes(frame, bbox= [boxes[2:6]], identities= boxes[1])
    cv2.waitKey(0)

    # Display the resulting frame
    frame = cv2.resize(frame, (640, 640)) 
    cv2.imshow('Frame', frame)
    

