# importing libraries
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('/home/ngoc/Documents/Yolov5_StrongSORT_OSNet/video_test/2022-10-05-09-30_cut.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
	print("Error opening video file")

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

result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (640,640))


i = 0
while(cap.isOpened()):
    cap.set(cv2.CAP_PROP_FPS,10)
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        i +=1
        print(i)
        for boxes in data:
            if boxes[0] == str(i):
                # Draw box
                frame = draw_boxes(frame, bbox= [boxes[2:6]], identities= boxes[1])
    
        
        # Display the resulting frame
        frame = cv2.resize(frame, (640, 640)) 
        result.write(frame)
        cv2.imshow('Frame', frame)
        
        # Press Q on keyboard to exit
        # cv2.waitKey(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loopqq
    else:
        break

# When everything done, release
# the video capture object
cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()