import argparse
parser = argparse.ArgumentParser(description='Load Video, run mrcnn, save results')
parser.add_argument('data', type=str, help='Name of Dataset, e.g.: LUEBECK', default='LUEBECK')
parser.add_argument('mode', type=bool, help='True: save video frome data, False: Make data from video', default=False)
args = parser.parse_args()
print(args.data)
print(args.mode)

import warnings
import os
import sys
import random
import math
import glob
import numpy as np
#import skimage.io
import imageio
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import imageio
from pathlib import Path
#import imgaug
import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Root directory of the project

#--------------------DIRECTORY--------------------
#DATASET = 'LUEBECK/'
DATASET = args.data
MODE = False

DIR = Path(os.path.dirname(os.path.realpath('__file__')))


if (DIR == Path('/beegfs/home/users/t/tim.schroeder')):
    CODE_DIR = os.path.join(DIR, 'object_rep/project_code/') 
else: 
    CODE_DIR = DIR

PROJECT_DIR = Path(CODE_DIR).parent

DATA_DIR = os.path.join(PROJECT_DIR, 'project_data/', DATASET)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'project_results/',DATASET)
if os.path.exists(RESULTS_DIR) == False:
    os.mkdir(RESULTS_DIR)

ROOT_DIR = os.path.join(CODE_DIR, "Mask_RCNN-master/")

print('DATASET: %s' % DATASET)
print('DIR: %s' % DIR)
print('CODE_DIR: %s' % CODE_DIR)
print('PROJECT_DIR: %s' % PROJECT_DIR)
print('DATA_DIR: %s' % DATA_DIR)
print('RESULTS_DIR: %s' % RESULTS_DIR)
print('ROOT_DIR: %s' % ROOT_DIR)

#--------------------COCO--------------------
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush', 'BG_object_close']


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import save_image
from mrcnn.visualize import return_image

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    if DATASET == 'LUEBECK':
        IMAGE_MIN_DIM = 768
        IMAGE_MAX_DIM = 1088

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

#--------------------FUNCTIONS--------------------

import ntpath
def path_leaf(path): # Get file name from path
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# Load Results when there are non
def save_detect_video(v_name, path, maxframes):
    """
    print('detect_img for video: %s' % v_name)
    videoloc = os.path.join(DATA_DIR, v_name, '*')
    vid = imageio.get_reader(path,  'ffmpeg')
    count = 0
    video_mask = []
    detect_img = []

    for image in vid.iter_data():
        if count <= maxframes:
            results = model.detect([image], verbose=1)
            detect_img.append(results[0])
            count += 1
        else:
            break
    
    count = 0
    detect_img = []
    print('load frames for video: %s' % v_name)
    vidlist = read_video(path)
    for image in vidlist:
        if count <= maxframes:
            results = model.detect([image], verbose=1)
            detect_img.append(results[0])
            count += 1
        else:
            print('maxframes break')
            break
    """
    detect_img = []
    cap = cv2.VideoCapture(path)
    success,image = cap.read()
    count = 0
    success = True
    while success:
        success,image = cap.read()
        results = model.detect([image], verbose=1)
        detect_img.append(results[0])
        count += 1
    # save to npz file (compressed)
    np.savez_compressed((RESULTS_DIR + '/test_detect_img_' + v_name), detect_img=detect_img)
    del detect_img
    cap.release(); del cap
    cv2.destroyAllWindows()

def load_results(PATH): 
    loadlib = np.load(PATH, allow_pickle=True) 
    results = loadlib['detect_img']
    return results

def read_video(path):
    vidlist = []
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if (cap.isOpened()== False): 
        print("Error opening video stream or file: %s" % path)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            vidlist.append(frame)
        else:
            break 

    cap.release()
    cv2.destroyAllWindows()
    return vidlist

def main():
    #save_video = MODE #save_video = TRUE: load results, make video; save_video = FALSE make results, save results
    
    vid_path_list = [video for video in glob.glob(DATA_DIR + '/*')]
    vid_path_list.sort()

    videos_run = [s for s in range(len(vid_path_list))]
    maxframes = 10
    fps=30
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    for v in range(len(vid_path_list)):
        v_name = os.path.splitext(path_leaf(vid_path_list[v]))[0]
        print('Video: %s' % v_name)
        
        vidlist = [] 
        vidlist = read_video(vid_path_list[v])
        
        if MODE == True:
            PATH = (RESULTS_DIR + f'/detect_img_' + v_name  + '.npz') 
            if os.path.exists(PATH):
                print('load video')
                results = []
                results = load_results(PATH) #load if they are in RESULT_DIR, otherwise run MRCNN

                frame_width = vidlist[v].shape[1]
                frame_height = vidlist[v].shape[0]
                print('load video')
                #Write Video
                out = cv2.VideoWriter(RESULTS_DIR + 'results_' + v_name +'.AVI' ,cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width,frame_height))

                for i in range(min(len(vidlist),maxframes)):
                    r = results[i]
                    img = (vidlist[i])
                    write_frame = np.asarray(return_image(img,i,r['rois'], r['masks'], r['class_ids'], r['scores'], class_names))
                    out.write(write_frame)
                
                out.release()
                cv2.destroyAllWindows()
            
            else: 
                print('no detect_img file found for video: %s' % PATH)
                
        else:
            save_detect_video(v_name, vid_path_list[v], maxframes) #run MRCNN
            


main()