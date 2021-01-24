# Densely Deformable Efficient Salient Object Detection Network

### Contents
1. Introduction
2. The proposed DDNet
3. Requirements and how to run?

## Introduction <hr>
In this paper, inspired by the best background=foreground separation abilities of deformable convolutions, we employ them in our Densely Deformable Network
(DDNet) to achieve efficient SOD. The salient regions from densely deformable convolutions are
further refined using transposed convolutions to optimally generate the saliency maps. Quantitative
and qualitative evaluations using the recent SOD
dataset against 22 competing techniques show our
method’s efficiency and effectiveness.

## The proposed DDNet <hr>
The proposed DDNet uses three main blocks to generate optimal saliency. Firstly, two dense convolution blocks represent lowlevel features of the input RGB images. Then we propose densely connected deformable convolutions to learn effective features of salient regions and their corresponding boundaries. Finally, we employ transpose convolution and upsampling to generate the resultant saliency image, refer to the figure below:


## This research is under process and once accepted, the codes and related data will be uploaded to this repository.
### Sorry for the inconvenience.
