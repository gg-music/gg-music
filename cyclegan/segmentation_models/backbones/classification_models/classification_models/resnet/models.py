from .builder import build_resnet
from ..utils import load_model_weights
from ..weights import weights_collection


def ResNet18(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    name = 'resnet18'
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(2, 2, 2, 2),
                         classes=classes,
                         include_top=include_top,
                         block_type='basic',
                         name=name)


    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNet34(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    name = 'resnet34'
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=include_top,
                         block_type='basic',
                         name=name)


    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNet50(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    name = 'resnet50'
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=include_top,
                         name=name)


    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNet101(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    name = 'resnet101'
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 4, 23, 3),
                         classes=classes,
                         include_top=include_top,
                         name=name)


    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def ResNet152(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    name = 'resnet152'
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 8, 36, 3),
                         classes=classes,
                         include_top=include_top,
                         name=name)


    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model
