#!/usr/bin/env python
import os 
import cv2 
import rospy 
import numpy as np
import torch 
from std_msgs.msg import String 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError
from model import ModelNetwork
from torch import optim
from torch import nn
import matplotlib.pyplot as plt

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ModelNetwork()#.to(device)

checkpoint = torch.load(os.path.join('../checkpoint/SFANet', 'checkpoint_latest.pth'))
model.load_state_dict(checkpoint['model'])
model.eval()


bridge = CvBridge() 
		
def preprocess_image(image):
    resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    resized_image = np.resize(resized_image, (1, 3, IMAGE_WIDTH, IMAGE_HEIGHT))
    resized_image = torch.from_numpy(resized_image)
    resized_image = resized_image.type(torch.float32)
    return resized_image

def image_cb(data):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = preprocess_image(cv_image)
    prediction, _ = model(image)
    count = prediction.sum().item()
    return count 

def main():
    rospy.init_node('counter', anonymous=True)
    image_subscriber = rospy.Subscriber('/webcam/image_raw', Image, image_cb, queue_size=1, buff_size=10000) #/camera/rgb/image_raw for real world
    try: 
        rospy.spin()
    except KeyboardInterrupt as e:
        print("shutting Down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()