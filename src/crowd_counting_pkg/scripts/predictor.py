import imp
import os 
import cv2 
import rospy 
import numpy as np
import torch 
from std_msgs.msg import String 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError
from models.SAFNet.src.utils.transforms import Transforms

bridge = CvBridge() 
model = torch.jit.load('../models/SAFNet/checkpoint/SFANet/model_scripted.pt')
model.eval() 

def image_cb(data):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BRG2RGB)
    image = Transforms(cv_image)
    prediction, _ = model(image)
    return prediction 

def main():
    rospy.init_node('counter', anonymous=True)
    image_subscriber = rospy.Subscriber('/webcam/image_raw', Image, image_cb, queue_size=10, buff_size=10000)
    try: 
        rospy.spin()
    except KeyboardInterrupt as e:
        print("shutting Down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()