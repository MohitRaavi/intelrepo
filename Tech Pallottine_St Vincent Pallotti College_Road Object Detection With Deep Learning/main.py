# importing required libraries

import torch
import cv2
import time
import matplotlib.pyplot as plot_boxes
from datetime import datetime

c=0
# Main fucntion

def main(image_path = None, video_path = None, video_out = None,webcam=None):

    print("Loading YOLOv7 model . . . ")
    ## loading our custom yolov5 trained model
    model =  torch.hub.load("/home/sky/Desktop/pothole_Object _Det/Detection/yolov7", 'custom', source ='local', path_or_model='/home/sky/Desktop/pothole_Object _Det/Detection/models/best.pt', force_reload=True) ### The model is stored locally
    class_names = model.names ### class names in string format
    # print(class_names)        

    model_obj =  torch.hub.load("/home/sky/Desktop/pothole_Object _Det/Detection/yolov7/", 'custom', source ='local', path_or_model='/home/sky/Desktop/pothole_Object _Det/Detection/models/yolov7.pt', force_reload=True) ### The model is stored locally
    class_names_obj = model_obj.names ### class names in string format
    # print(class_names_obj)        


    if video_path or webcam:
        print("Working with Video File . . .")

        ## reading the video
        if webcam:
            cap = cv2.VideoCapture(0)
        elif video_path:
            cap=cv2.VideoCapture(video_path)
            frames = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
            # print("Total Frames=",frames)
            fps_count = int(cap.get(cv2.CAP_PROP_FPS))
        
        if video_out: # creating the video writer if video output path is given
            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v') ##('XVID')
            out = cv2.VideoWriter(video_out, codec, 20, (width, height))

        # # print(width, height)
        # print("FPS:",fps)
        

        # assert cap.isOpened()
        frame_no = 1

        # cv2.namedWindow("Video_Capture", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret:
                print("Working on frame by frame video file . . .")
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                result_pothole = model_pred(frame, model = model)
                result_object = model_pred(frame, model = model_obj)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                if webcam:
                    frame = plot_boxes(result_pothole, frame,class_names,width, height)
                    frame = plot_boxes(result_object, frame,class_names_obj,width, height)
                elif video_path:
                    frame = plot_boxes(result_pothole, frame,class_names,width, height,fps=fps_count,frame_no=frame_no)
                    frame = plot_boxes(result_object, frame,class_names_obj,width, height)
                cv2.imshow("Captured Video", frame)
                if video_out:
                    print("Saving out predicted output video . . .")
                    out.write(frame)
                # if webcam:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break  
            frame_no += 1
        print("Present Frame:",frame_no)
        print('Exiting from all the windows . . .')    
        cap.release()
        # closing all available windows
        cv2.destroyAllWindows()


def model_pred (frame, model):
    # print(frame)
    # print("Sit tight, Your work is in progress...")
    results = model(frame)
    # results.show() # This will display your output
    # print(results.xyxyn)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    # print(labels, cordinates)

    return labels, cordinates


def plot_boxes(results, frame,classes,width,height,fps=None,frame_no=None):
    labels, img_cords = results
    #print(labels, img_cords)
    n_detections = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    # print("Total Number of Detections made:", n_detections)
    # print("Looping Through the Detections . . .")
    area_img=width*height

    # looping through the detections
    c=0
    for i in range(n_detections):
        bbox_cords = img_cords[i]
        # print(bbox_cords)
        if round(float(bbox_cords[4]),2) > 0.3: # threshold value for detection. bbox_cords[4] is the confidence of prediction and we discard every detection with confidence less than 0.7.
            x1, y1, x2, y2 = int(bbox_cords[0]*x_shape), int(bbox_cords[1]*y_shape), int(bbox_cords[2]*x_shape), int(bbox_cords[3]*y_shape) ## BBOx coordniates
            label_name = classes[int(labels[i])]            
            # print(x1, y1, x2, y2)
            w_frame=x2-x1
            h_frame=y2-y1
            # label_name = 'pothole Left'
            # print(label_name)
            area=round(float(w_frame*h_frame*100/area_img),2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## Drawing Bouding Boxes
            #cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## Bounding Box for label
            # cv2.putText(frame, label_name + f" {round(float(bbox_cords[4]),2)}"+ " Area:"+ f" {area}"+"%", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
            cv2.putText(frame, label_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
    
    return frame



# calling main function to run the program
main(video_path=r"1.mp4", video_out="result.mp4")
# main(webcam=True, video_out="demo_result.mp4") # Activate this fucntion when you need to capture output from webcam
# main(image_path="train_images/1.jpg") # Activate this fucntion when you need to capture output from image