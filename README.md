## Hypercolumns

Many algorithms using features from CNNs (Convolutional Neural Networks) usually use the last FC (fully-connected) layer features in order to extract information about certain input. However, the information in the last FC layer may be too coarse spatially to allow precise localization (due to sequences of maxpooling, etc.), on the other side, the first layers may be spatially precise but will lack semantic information. 

To get the best of both worlds, the authors of the hypercolumn paper define the hypercolumn of a pixel as the vector of activations of all CNN units “above” that pixel.

![Hypercolumns-image](http://blog.christianperone.com/wp-content/uploads/2016/01/hypercolumn.png)


## Acknowledgments

[Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/abs/1411.5752)

[Convolutional hypercolumns in Python](http://blog.christianperone.com/2016/01/convolutional-hypercolumns-in-python/)