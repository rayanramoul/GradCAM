import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pickle
import matplotlib.patches as patches
import cv2
import numpy as np
import random

"""
This is a script to visualize 10 random images and their ground truth + predicted bounding boxes
"""


if __name__ == '__main__':
    # Avaiable thresholds data
    thresholds = [0.01, 0.02, 0.06] 
    threshold = thresholds[2]

    # Loading data
    with open('results/all_gt_dogs.pickle', 'rb') as handle:
        all_gt_dogs = pickle.load(handle)
    with open('results/all_detected_dogs'+str(threshold)+'.pickle', 'rb') as handle:
        all_detected_dogs = pickle.load(handle)

    # Get 10 random images
    for i in range(10):
        key = random.choice(list(all_detected_dogs.keys()))
        fig,ax = plt.subplots(1)
        img = cv2.imread(key, 1)
        img = np.float32(img) / 255
        img = img[:, :, ::-1]

        ax.imshow(img)
        gt = all_gt_dogs[key]
        predicted = all_detected_dogs[key]
        # Loop in objects of ground truth
        for obj in gt:
            xmingt, ymingt, xmaxgt, ymaxgt = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            widthgt = xmaxgt-xmingt
            heightgt = ymaxgt-ymingt
            rect0 = patches.Rectangle((xmingt,ymingt),widthgt,heightgt,linewidth=3,edgecolor='r',facecolor='none', label="predicted")
        
            centerxgt = xmingt + widthgt/2
            centerygt = ymingt + heightgt/2
            # Add the patch to the Axes
            ax.add_patch(rect0)
            
            plt.text(centerxgt, centerygt,'Ground Truth', bbox=dict(facecolor='red', alpha=0.5))
            
        # Loop in objects predicted
        for obj in predicted:
            xmin, ymin, xmax, ymax =  int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin,ymin),width,height,linewidth=3,edgecolor='b',facecolor='none', label="ground truth")
            
            centerx = xmin + width/2
            centery = ymin + height/2
            ax.add_patch(rect)
        plt.text(centerx, centery,'Predicted', bbox=dict(facecolor='blue', alpha=0.5))
        plt.show()
