# Tutorial-on-Implementation-of-YOLO-V3-in-Pytorch

## Overview 

Reading codes with little comments or help document could be a hugh headache especially for most new-entry deep learning reserach engineers. This repo is projected to offer a tutorial on how to implement YOLO_V3, one of the state of art deep learning algorithms for object detection. 

In this work, the YOLO_V3 algorithm is trained from stratch using Pascal VOC dataset for demonstration purpose. Hopefully, after reading this tutorial, developers can build and train their own YOLO network using other datasets for various object detection tasks



## Installation 

### Environment 
* pytorch >= 1.0.0
* python >= 3.6.0
* numpy
* opencv-python
* pip3 install -r requirements.txt

### download Pascal VOC Data 
follow the instruction from this [link](https://pjreddie.com/darknet/yolo/)

### download the weights
1. download the pretrained weights "Dartnet_VOC_Weights" and "Dartnet_VOC_Weights_ini" from Google Drive or Baidu Drive
2. Move downloaded both files to wegihts folder in this project.

## Inference 
* python Test.py --confidence 0.5 --reso 416
* Refer to jupyter notebook "Yolo_V3_Train_Step_by_Step" for instruction

## Training 
* python Train.py --epochs 25 --batch_size 16 --img_size 416
* Refer tp jupyter notebook "Yolo_V3_Train_Step_by_Step" for instruction 

## Reference 
* [YOLO:Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
* [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)




