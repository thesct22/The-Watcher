# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 03:44:03 2019

@author: Sharath C.Thomas
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:46:57 2019

@author: Sharath C.Thomas
"""


import argparse
import logging
import time
import requests
from urllib.request import urlopen
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from keras.models import load_model
model_humans = load_model('model_1.h5')
#import scripts.label_image as label_img
video=[]
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


#video="D:\DATASET\DCSASS Dataset\Shoplifting\Shoplifting001_x264.mp4\Shoplifting001_x264_3.mp4"
#video="D:\DATASET\DCSASS Dataset\Shoplifting\Shoplifting001_x264.mp4\Shoplifting001_x264_4.mp4"
#video="D:\DATASET\DCSASS Dataset\Shoplifting\Shoplifting001_x264.mp4\Shoplifting001_x264_5.mp4"
#video="D:\DATASET\DCSASS Dataset\Shoplifting\Shoplifting001_x264.mp4\Shoplifting001_x264_6.mp4"
video="D:\DATASET\DCSASS Dataset\Shoplifting\Shoplifting001_x264.mp4\Shoplifting001_x264_7.mp4"

#video="D:\DATASET\3\NonViolence\football_crowds__Inside_the_Iowa_Spirit_Squads__uimediaproduction__V365Mro2t88.avi"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    
            
#    video=
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
#    cap = cv2.VideoCapture(video)
#    if cap.isOpened() is False:
#        print("Error opening video stream or file")
#    while cap.isOpened():
#        ret_val, image = cap.read()cam = 
    cam=cv2.VideoCapture("http://192.168.43.45:8160/")
    #image variable image
    ret_val, image = cam.read()
#      logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    

    count = 0
    while True:
        per1=0
        
        logger.debug('+image processing+')
        ret_val, image = cam.read()
        
        logger.debug('+postprocessing+')

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        Sum=0
        if len(humans)>0:
            for human in humans:
                x=[]
                for i in range(15):
                    try:
                        x.append(human.body_parts[i].x)
                        x.append(human.body_parts[i].y)
                    except:
                        x.append(0)
                        x.append(0)
                x=[x]
                x=np.array(x)
                res=model_humans.predict(x)
                Sum=Sum+res[0][1]        
        else:
            x=[0]*30
            x=[x]
            x=np.array(x)
            res=model_humans.predict(x)
            
       
        
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        fps=video
#        if not len(fps):
        if len(humans)==0:
            per=Sum
        else:
            per=(Sum)/(len(humans))
        
        
        cv2.putText(image,"Suspicion: %f" % (per*100),(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
#        else:
#        cv2.putText(image, "Suspicion %f  percent"    % (res[0][0]*100), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        if (per>=0.60)or( (per-per1!=per)and(per-per1>0.20)):
            count=count+1
            print(count)
            if count%50==0 or count==0:
                print("Data sent to the mobile phone")
#                print("WE be have send",count)
                requests.post("https://maker.ifttt.com/trigger/shop_lifting/with/key/dFjEYVUodU_mx8eGJ98d58",params={"value1":"none","value2":"none","value3":"none"})
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')

