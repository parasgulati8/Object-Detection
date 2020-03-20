# Object Detection
 Object Detection using Yolo algorithm
## 1 - Problem Statement

The project aims to detect various objects and draw a bounding box around them.

## 2 - YOLO

YOLO ("you only look once") is a popular algoritm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

The **input** is a batch of images of shape (m, 608, 608, 3) and the **output** is a list of bounding boxes along with the recognized classes.

We used 5 anchor boxes. So you can think of the YOLO architecture as the following: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

One way to visualize what YOLO is predicting on an image:
- For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across both the 5 anchor boxes and across different classes). 
- Color that grid cell according to what object that grid cell considers the most likely.

Doing this results in this picture: 

<img src="nb_images/proba_map.png" style="width:300px;height:300;">

Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:  

<img src="nb_images/anchor_map.png" style="width:200px;height:200;">
In total, the model predicts: 19x19x5 = 1805 boxes just by looking once at the image (one forward pass through the network)! Different colors denote different classes.

In the figure above, we plotted only boxes that the model had assigned a high probability to, but this is still too many boxes. To filter the algorithm's output we use non-max suppression.

### 2.2 - Filtering with a threshold on class scores

We would like to get rid of any box for which the class "score" is less than a chosen threshold. 

For each box, we find:
    - the index of the class with the maximum box score
    - the corresponding box score 

We create a mask by using a threshold. For example : `([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4)` returns: `[False, True, False, False, True]`. The mask should be True for the boxes you want to keep. 

TensorFlow is used to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes we don't want. We are left with just the subset of boxes we want to keep.

### 2.3 - Non-max suppression ###

Even after filtering by thresholding over the classes scores, there will still remain a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS). 

<img src="nb_images/non-max-suppression.png" style="width:500px;height:400;">
<caption><center> <u> **Figure 7** </u>: In this example, the model has predicted 3 cars, but it's actually 3 predictions of the same car. Running non-max suppression (NMS) will select only the most accurate (highest probabiliy) one of the 3 boxes. <br> </center></caption>

The key steps for NMS are: 
1. Select the box that has the highest score.
2. Compute its overlap with all other boxes, and remove boxes that overlap it more than `iou_threshold`.
3. Go back to step 1 and iterate until there's no more boxes with a lower score than the current selected box.

This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.

### 2.4 Wrapping up the filtering
`yolo_eval()` takes the output of the YOLO encoding and filters the boxes using score threshold and NMS.


## 3 - Test YOLO pretrained model on images
We are trying to detect 80 classes, and are using 5 anchor boxes. We have gathered the information about the 80 classes and 5 boxes in two files "coco_classes.txt" and "yolo_anchors.txt".
 
The car detection dataset has 720x1280 images, which we've pre-processed into 608x608 images.
 
Load an existing pretrained Keras YOLO model stored in "yolo.h5". (These weights come from the official YOLO website, and were converted using a function written by Allan Zelener. Technically, these are the parameters from the "YOLOv2" model)

This model converts a preprocessed batch of input images (shape: (m, 608, 608, 3)) into a tensor of shape (m, 19, 19, 5, 85) that needs to pass through non-trivial processing and conversion.

`yolo_outputs` gives all the predicted boxes of yolo_model in the correct format. To perform filtering and select only the best boxes, we call `yolo_eval`

The final output of the model is an image with all the objects bounded by rectangular boxes and also returns the prediction probabilities corresponding to each object detected. For example :

![](https://github.com/parasgulati8/Object-Detection/blob/master/out/test.jpg)
 
 ## Conclusion
 - YOLO is a state-of-the-art object detection model that is fast and accurate
- It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume. 
- The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
- You filter through all the boxes using non-max suppression. Specifically: 
    - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
    - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
- Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise. 

## References
**References**: The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's github repository. The pretrained weights used in this exercise came from the official YOLO website. 
- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
- Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
- Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
- The official YOLO website (https://pjreddie.com/darknet/yolo/) 

https://www.coursera.org/lecture/convolutional-neural-networks/yolo-algorithm-fF3O0
