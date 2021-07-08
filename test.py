import random
import matplotlib.pyplot as plt
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import sys
import warnings
import numpy as np
import cracks
import glob
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score , confusion_matrix
if not sys.warnoptions:
    warnings.simplefilter("ignore")

DATASET_DIR = r'datasets\crack_segmentation_dataset'

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

#CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
CLASS_NAMES = ['1','0']

file = open('outputs.txt','w')

class CrackConfig(mrcnn.config.Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "cracks"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cracks

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 413
    VALIDATION_STEPS = 45

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 50

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 50

    LEARNING_RATE = 0.0001


class CrackInferenceConfig(CrackConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7





def load_model():
    model = mrcnn.model.MaskRCNN(mode="inference",
                                 config=CrackInferenceConfig(),
                                 model_dir=os.getcwd())

    model.load_weights("logs\cracks20210701T2000\mask_rcnn_cracks_0030.h5", by_name=True)

    return model

def mean_IOU(gt,pred):
    conf_mat = confusion_matrix(y_true=gt,y_pred=pred)
    #print(conf_mat)

    MIOU= np.diag(conf_mat) / (np.sum(conf_mat,axis=1) +  np.sum(conf_mat,axis= 0) - np.diag(conf_mat))
    MIOU = np.nanmean(MIOU)
    return MIOU




def infer():
    model = load_model()
    ious = []
    path = 'crack_segmentation_dataset/test'
    imgs_path = glob.glob(path + '/images/*')
    masks_path = glob.glob(path + '/masks/*')
    count = 1
    for img_path, mask_path in zip(imgs_path, masks_path):
           try :
               img = cv2.imread(img_path)
               (thresh, mask) = cv2.threshold(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY), 128, 255,
                                              cv2.THRESH_BINARY)

               predicted = model.detect([img])[0]['masks'].astype(np.uint8).max(axis=-1)
               mask = mask.astype(np.uint8) // 255
               cv2.imshow('pred',predicted*255)
               cv2.imshow('mask',mask*255)
               cv2.waitKey(1)

               jac = jaccard_score(mask.flatten(), predicted.flatten())
               f1 = f1_score(predicted.flatten(),mask.flatten())
               mean_iou = mean_IOU(gt=mask.flatten(),pred=predicted.flatten())
               count += 1
               file.write("img_name : {} JAC : {} F1 : {} IOU : {}".format(str.split(img_path,sep='\\')[-1],jac,f1,mean_iou))
               file.write('\n')
               print("img_name : {} JAC : {} F1 : {} IOU : {}".format(str.split(img_path,sep='\\')[-1],jac,f1,mean_iou))
           except :
               print('exception occured')







infer()







# dir_img = "crack_segmentation_dataset\images"
# dir_masks = "crack_segmentation_dataset\masks"
# imgs =  glob.glob(dir_img + '/*')
# masks = glob.glob(dir_masks + '/*')
# idxs = random.sample(k=20,population=range(len(imgs)))
# i=0
# for idx in idxs:
#     i+=1
#     image = plt.imread(imgs[idx])
#     mask = plt.imread(masks[idx])
#     plt.imsave('logs\\mask {}.jpg'.format(i),mask)
#     r = model.detect([image])
#     r = r[0]
#     mrcnn.visualize.display_instances(image=image,
#                                   boxes=r['rois'],
#                                   masks=r['masks'],
#                                   class_ids=r['class_ids'],
#                                   class_names=CLASS_NAMES,
#                                   scores=r['scores'])
#     print('image {} saved'.format(i))
#     plt.imsave('logs\\images\\predicted {}.jpg'.format(i),img)


