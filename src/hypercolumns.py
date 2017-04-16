import theano
import cv2
import numpy as np
import scipy as sp

def extract_hypercolumn(model, layer_indexes, instance):
    layers = [model.layers[li].get_output(train=False) for li in layer_indexes]
    get_feature = theano.function([model.layers[0].input], layers,
                                  allow_input_downcast=False)
    feature_maps = get_feature(instance)
    hypercolumns = []    
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = sp.misc.imresize(fmap, size=(224, 224),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)



layers_extract = [3, 8]
hc = extract_hypercolumn(model, layers_extract, im)    