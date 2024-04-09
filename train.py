import sklearn
import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
imgs_path='imgs/0'

for img_name in os.listdir(imgs_path):
    img=Image.open(os.path.join(imgs_path,img_name))
    img=np.array(img)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.threshold(img,100,255,cv2.THRESH_BINARY)[1]


    #plt.imshow(img)
    #plt.show()