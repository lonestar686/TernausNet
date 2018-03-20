
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.utils.data_utils import get_file
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': {'with_top':'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
              'with_notop':'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'}
    'vgg19': {'with_top':'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5',
              'with_notop':'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'}
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(object):

    def __init__(self, features, num_classes=1000, p=0.25):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = Sequential([
            Flatten(),
            Dense(4096, activation='relu'),   # from (512 * 7 * 7)
            Dropout(p),
            Dense(4096, activation='relu'),
            Dropout(p),
            Dense(num_classes, activation='softmax')
        ])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, input_shape, batch_norm=False):
    """ build VGG feature layers

    """
    input = Input(shape=input_shape)    # in_channels = 3

    layers = [input]
    for v in cfg:
        if v == 'M':
            layers += [MaxPooling2D(pool_size=2 strides=2, padding='same')]
        else:
            conv2d = Conv2D(v, kernel_size=3, padding='same')
            if batch_norm:
                layers += [conv2d, BatchNormalization(), Activation('relu')]
            else:
                layers += [conv2d, Activation('relu')]

    return Sequential(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(input_shape, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        input_shape:   input shape (224x224x3)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], input_shape), **kwargs)
    if pretrained:
        model.load_weights(model_urls['vgg11'])
    return model


def vgg11_bn(input_shape, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        input_shape:   input shape (224x224x3)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], input_shape, batch_norm=True), **kwargs)
    if pretrained:
        model.load_weights(model_urls['vgg11_bn'])
    return model


def vgg13(input_shape, pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        input_shape:   input shape (224x224x3)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], input_shape), **kwargs)
    if pretrained:
        model.load_weights(model_urls['vgg13'])
    return model


def vgg13_bn(input_shape, pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        input_shape:   input shape (224x224x3)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], input_shape, batch_norm=True), **kwargs)
    if pretrained:
        model.load_weights(model_urls['vgg13_bn'])
    return model


def vgg16(input_shape, include_top=False, weights='imagenet', **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        input_shape:   input shape (224x224x3)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], input_shape), **kwargs)

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    model_urls['vgg16']['with_top'],
                                    cache_subdir='models',
                                    file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    model_urls['vgg16']['with_notop'],
                                    cache_subdir='models',
                                    file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)

    return model


def vgg16_bn(input_shape, pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        input_shape:   input shape (224x224x3)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], input_shape, batch_norm=True), **kwargs)
    if pretrained:
        model.load_weights(model_urls['vgg16_bn'])
    return model


def vgg19(input_shape, include_top=False, weights='imagenet', **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        input_shape:   input shape (224x224x3)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], input_shape), **kwargs)

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    model_urls['vgg19']['with_top'],
                                    cache_subdir='models',
                                    file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    model_urls['vgg19']['with_top'],
                                    cache_subdir='models',
                                    file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)

    return model


def vgg19_bn(input_shape, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        input_shape:   input shape (224x224x3)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], input_shape, batch_norm=True), **kwargs)
    if pretrained:
        model.load_weights(model_urls['vgg19_bn'])
    return model