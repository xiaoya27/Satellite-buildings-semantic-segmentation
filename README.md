# Satellite-buildings-semantic-segmentation
Data source: https://www.kaggle.com/hyyyrwang/buildings-dataset


## Neptune Experiment tracking

https://ui.neptune.ai/xiaoya27/BuildingSegmentation/experiments?viewId=standard-view


## Experiment design 

1. FCN vs CNN
![img](https://drive.google.com/uc?export=view&id=1lrqB4YegV8jXWNfyYAaeuFlwXIc54aRP)

Fully convolutional indicates that the neural network is composed of convolutional layers without any fully-connected layers 

2. FCN vs U-net
Compared to FCN-8, the two main differences are 

(1) U-net is symmetric 

(2) the skip connection between the down-sampling path and the up-sampling path apply a concatenation operator instead of a sum.

3. Unet vs FPN Difference


Unet

![img](https://miro.medium.com/max/770/1*O2NbipwBOdTMtj7ThBNTPQ.png)


FPN

![img](https://miro.medium.com/max/759/1*D_EAjMnlR9v4LqHhEYZJLg.png)


FPN: ([ref](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610))

1. the spatial resolution is upsampled by a factor of 2 using the nearest neighbor for simplicity.
2. the feature maps from bottom-up pathway undergoes 1×1 convolutions to reduce the channel dimensions.
3. And the feature maps from the bottom-up pathway and the top-down pathway are merged by element-wise addition.



BackBone Model : EfficientNet ----why it is more efficient?
[ref](https://heartbeat.fritz.ai/reviewing-efficientnet-increasing-the-accuracy-and-robustness-of-cnns-6aaf411fc81d)


1. Depthwise Convolution + Pointwise Convolution: Divides the original convolution into two stages to significantly reduce the cost of calculation, with a minimum loss of accuracy.
2. Inverse Res(?): The original ResNet blocks consist of a layer that squeezes the channels, then a layer that extends the channels. In this way, it links skip connections to rich 
3. channel layers. In MBConv, however, blocks consist of a layer that first extends channels and then compresses them, so that layers with fewer channels are skip connected.
Linear bottleneck: Uses linear activation in the last layer in each block to prevent loss of information from ReLU.


The main building block for EfficientNet is MBConv, an inverted bottleneck conv, originally known as MobileNetV2. Using shortcuts between bottlenecks by connecting a much smaller number of channels (compared to expansion layers), it was combined with an in-depth separable convolution, which reduced the calculation by almost k² compared to traditional layers. Where k denotes the kernel size, it specifies the height and width of the 2-dimensional convolution window.


    from keras.layers import Conv2D, DepthwiseConv2D, Adddef inverted_residual_block(x, expand=64, squeeze=16):

    block = Conv2D(expand, (1,1), activation=’relu’)(x)
    
    block = DepthwiseConv2D((3,3), activation=’relu’)(block)
    
    block = Conv2D(squeeze, (1,1), activation=’relu’)(block)
    
    return Add()([block, x])
    



TODO: 
1. Replace the DataLoader with high performance data pipelines using tf.data 

Cuz : Warning: Tensorflow:multiprocessing can interact badly with TensorFlow, causing nondeterministic deadlocks. For high performance data pipelines tf.data is recommended.

2. FCN-8 + EfficientNet as encoder
