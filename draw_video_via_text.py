# importing libraries
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('video_test/origin_video.mp4')

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
with open ("origin_video.txt", "r") as f:
    f = f.read()
    f = f.split("\n")

for i in f:
    i = i.split(" ")
    data.append(i)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (1036, 700))


#Dữ liệu để theo dõi quỹ đạo
trajectory  = []
current_centroidarr_x = 0
current_centroidarr_y = 0
previous_centroidarr_x = 0
previous_centroidarr_y = 0

i = 0
while(cap.isOpened()):
    cap.set(cv2.CAP_PROP_FPS,10)
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        i +=1
        print(i)
        for boxes in data:
            #Kiểm tra nếu Frame thứ i trong file txt trùng với Frame thứ i trong video đang đọc
            if boxes[0] == str(i):
                # Draw box
                frame = draw_boxes(frame, bbox= [boxes[2:6]], identities= boxes[1])

                #Vẽ quỹ đạo cho video
                trajectory.append(boxes)
                # print(trajectory)
                # print(trajectory)
                
                if len(trajectory) >1:
                    if len(trajectory)>40:
                        trajectory = trajectory[-40:]
                    # Từng lịch sử frame 1
                    for i_tjtr in range(len(trajectory)-1,0,-1):
                        # if i_tjtr == 0:
                        #     continue 
                        print(trajectory[i_tjtr])
                        print("############################################")
                        #so sánh khung trước với khung sau
                        # for tra_crr in trajectory[i_tjtr]:
                        #     for tra_pre in trajectory[i_tjtr-1]:
                                # print(tra_pre, tra_crr)
                                # if tra_pre[1] == tra_crr[1]:
                                #     # print(tra_pre[4])
                                #     current_centroidarr_x = int(tra_crr[0] + tra_crr[2]/2)
                                #     current_centroidarr_y = int(tra_crr[1] + tra_crr[3]/2)
                                #     # print(current_centroidarr_x, current_centroidarr_y)

                                #     previous_centroidarr_x = int(tra_pre[0] + tra_pre[2]/2)
                                #     previous_centroidarr_y = int(tra_pre[1] + tra_pre[3]/2)
                                #     # print(previous_centroidarr_x, previous_centroidarr_y)


                                #     color = compute_color_for_labels(int(tra_pre[1]))

                                #     cv2.line(frame, (current_centroidarr_x,current_centroidarr_y),
                                #                 (previous_centroidarr_x,previous_centroidarr_y),
                                #                 color, thickness=3)   
        
        # Đặt lại kích cỡ video
        frame = cv2.resize(frame, (1036, 700)) 

        #Lưu và show video
        output_video.write(frame)
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
output_video.release()

# Closes all the frames
cv2.destroyAllWindows()