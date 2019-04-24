from __future__ import division
import time
import torch
from torch.autograd import Variable
import cv2
from utils.util import *
from Darknet_VOC import Darknet
import random
import argparse
import pickle as pkl
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def write(x, img):

    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    return img

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()


if __name__ == '__main__':

    classes = load_classes("data/voc.names")
    colors = pkl.load(open("data/pallete", "rb"))

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    CUDA = torch.cuda.is_available()

    num_classes = 20

    model = Darknet()
    print("Loading network.....")
    model.load_state_dict(torch.load('weights/Dartnet_VOC_Weights', map_location=lambda storage, loc: storage))
    print("Network successfully loaded")
    model.image_size = args.reso
    inp_dim = int(model.image_size)

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    start = time.time()

    frame = cv2.imread('imgs/timg.jpeg')
    img, orig_im = prep_image(frame, inp_dim)

    if CUDA:
        img = img.cuda()

    output, _ , _, _= model(Variable(img), CUDA)
    # batch_size x number of boxes x attrs (85)  attrs have been transposed to input image
    output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)
    # D x 8, D is the true detection  8: image index in batch, 4 corner coordinates, object score,highest class score, class index

    if isinstance(output, int) == False:
        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
        output[:, [1, 3]] *= orig_im.shape[1]
        output[:, [2, 4]] *= orig_im.shape[0]

        list(map(lambda x: write(x, orig_im), output))

    print("image predicted in {:2.3f} seconds".format(time.time()- start))

    cv2.imshow("frame", orig_im)
    cv2.waitKey(0)
