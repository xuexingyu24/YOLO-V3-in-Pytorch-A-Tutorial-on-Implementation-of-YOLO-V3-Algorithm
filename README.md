# Tutorial on Implementation of YOLO V3 in Pytorch

## Overview 

Reading codes with little comments could be a hugh headache especially for most new-entry machine learning reserach engineers. This repo is intended to offer a tutorial on how to implement YOLO V3, one of the state of art deep learning algorithms for object detection. 

In this work, the YOLO_V3 algorithm is trained from stratch using Pascal VOC dataset for demonstration purpose. Hopefully, after reading this tutorial, developers can build and train their own YOLO network using other datasets for various object detection tasks

### Example 

<img src = "https://github.com/xuexingyu24/Tutorial-on-Implementation-of-YOLO-V3-in-Pytorch/blob/master/imgs/timg_2.jpeg"  width="400" > <img src = "https://github.com/xuexingyu24/Tutorial-on-Implementation-of-YOLO-V3-in-Pytorch/blob/master/imgs/person_2.jpg"  width="400" >

## Requirement  

### Environment 
* pytorch >= 1.0.0
* python >= 3.6.0
* numpy
* opencv-python

### download Pascal VOC Data 
1. follow the instruction from [Yolo website](https://pjreddie.com/darknet/yolo/) or find the data from [link](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
2. download the voc_label.py script to genetrate label files 
  wget https://pjreddie.com/media/files/voc_label.py
  python voc_label.py

### download the weights
1. download the pretrained weights "Dartnet_VOC_Weights" and "Dartnet_VOC_Weights_ini" from [Baidu Drive](https://pan.baidu.com/s/1-O-jD0uU3OM6yNaUSLjAhw)
2. Move downloaded both files to weights folder in this project.

## Inference 
* python Test.py --confidence 0.5 --reso 416
* Refer to jupyter notebook [Yolo_V3_Inference_Step_by_Step](https://github.com/xuexingyu24/Tutorial-on-Implementation-of-YOLO-V3-in-Pytorch/blob/master/Yolo_V3_Inference_Step_by_Step.ipynb) for detailed instruction

## Training 
* python Train.py --epochs 25 --batch_size 16 --img_size 416
* Refer tp jupyter notebook [Yolo_V3_Train_Step_by_Step](https://github.com/xuexingyu24/Tutorial-on-Implementation-of-YOLO-V3-in-Pytorch/blob/master/Yolo_V3_Train_Step_by_Step.ipynb) for detailed instruction 

## Reference 
* [YOLO:Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
* [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
* YOLO tutorial (https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/): highly inspired by this post



