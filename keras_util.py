import numpy as np
from PIL import Image as pil_image
from math import floor
from keras import Model
from keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_preprocess
from keras.applications.nasnet import NASNetMobile
from keras.applications.densenet import DenseNet, preprocess_input as densenet_preprocess
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocess
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inceptionresnetv2_preprocess
from keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_preprocess
from keras.layers import GlobalAveragePooling2D, Lambda, Concatenate
import tensorflow as tf


def crop_load_all(filenames, load_size_range=(225, 325), crop_size=224, crops_per_image=1, grayscale=False, random_crop=False):
    n_images = len(filenames)
    n_channels = 3 if not grayscale else 1
    images = np.zeros((n_images * crops_per_image, crop_size, crop_size, n_channels), dtype=np.uint8)
    for i, filename in enumerate(filenames):
        for j in range(crops_per_image):
            load_size = np.random.randint(load_size_range[0], load_size_range[1] + 1)
            images[i * crops_per_image + j, :, :, :] = load_crop_img(
                filename,
                load_size=load_size,
                crop_size=crop_size,
                grayscale=grayscale,
                random=random_crop
            )
    return images


def batch_transform(loader, features, x, logger, batch_size=96):
    n_samples = x.shape[0]
    ratio = int(n_samples / batch_size)
    n_batches = int(ratio) if n_samples % batch_size == 0 else (int(ratio) + 1)
    n_features = np.prod(features._model.layers[-1].output_shape[1:])
    x_trans = np.zeros((x.shape[0], n_features), dtype=np.float32)
    for i in logger.monitor(range(n_batches), "Features extraction with batch_size '{}'".format(batch_size)):
        start = i * batch_size
        end = min(x.shape[0], start + batch_size)
        imgs = loader.transform(x[start:end])
        x_trans[start:end, :] = features.transform(imgs)
        del imgs
    return x_trans


def load_crop_img(path, load_size, crop_size, grayscale=False, random=True):
    """Loads an image into PIL format and crop it at a given size.

    # Arguments
        path: Path to image file
        load_size: int or (int, int),
            If integer, size the smallest side should be resized to. Otherwise the (height, width) to which the
            image should be resized to
        crop_size: int or (int, int),
            Size of the random square crop to be taken in the image (should be less then or equal to load_size)
            If a tuple is given the random crop is a rectangle of given (height, width)
        grayscale: Boolean, whether to load the image as grayscale.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')

    # load resize
    width, height = img.size
    if isinstance(load_size, tuple):
        new_height, new_width = load_size
    elif height < width:
        ratio = float(load_size) / height
        new_width, new_height = int(floor(ratio * width)), load_size
    elif width < height:
        ratio = float(load_size) / width
        new_width, new_height = load_size, int(floor(ratio * height))
    else:
        new_height, new_width = load_size, load_size

    img = img.resize((new_width, new_height))

    # (random) crop resize
    if isinstance(crop_size, tuple):
        crop_size_h, crop_size_w = crop_size
    else:
        crop_size_h, crop_size_w = crop_size, crop_size
    width, height = img.size
    len_crop_h, len_crop_w = height - crop_size_h, width - crop_size_w
    if random:
        offset_h = np.random.randint(len_crop_h + 1)
        offset_w = np.random.randint(len_crop_w + 1)
    else:
        offset_h = int(len_crop_h / 2)
        offset_w = int(len_crop_w / 2)
    return img.crop((offset_w, offset_h, offset_w + crop_size_w, offset_h + crop_size_h))


class ImageLoader(object):
    def __init__(self, load_size_range=(225, 325), crop_size=224, n_crops_per_image=1, random_crop=False):
        self._load_size_range = load_size_range
        self._crop_size = crop_size
        self._n_crops_per_image = n_crops_per_image
        self._random_crop = random_crop

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return crop_load_all(
            X, load_size_range=self._load_size_range,
            crop_size=self._crop_size,
            crops_per_image=self._n_crops_per_image,
            random_crop=self._random_crop
        )


MODEL_RESNET50 = "resnet50"
MODEL_VGG19 = "vgg19"
MODEL_VGG16 = "vgg16"
MODEL_INCEPTION_V3 = "inception_v3"
MODEL_INCEPTION_RESNET_V2 = "inception_resnet_v2"
MODEL_MOBILE = "mobile"
MODEL_DENSE_NET_201 = "dense_net_201"
MODEL_NASNET_LARGE = "nas_net_large"
MODEL_NASNET_MOBILE = "nas_net_mobile"


class PretrainedModelFeatures(object):
    def __init__(self, model=MODEL_RESNET50, layer="last", reduction="avg", weights="imagenet", filters=None):
        self._model_name = model
        self._weights = weights
        self._layer = layer
        self._reduction = reduction
        self._filters = filters
        self._model = self._get_model(
            self._model_name,
            layer=self._layer,
            reduction=self._reduction,
            weights=self._weights,
            filters=self._filters
        )

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        X = self._get_preprocessing(self._model_name)(X.astype(np.float))
        features = self._model.predict(X)
        return features.reshape((X.shape[0], -1))

    def __setstate__(self, state):
        self.__dict__ = state
        self._model = self._get_model(
            self._model_name,
            layer=self._layer,
            reduction=self._reduction,
            weights=self._weights
        )

    def __getstate__(self):
        self._model = None
        return self.__dict__

    @classmethod
    def _get_model(cls, model_name=None, layer="last", reduction="average_pooling", weights="imagenet", filters=None):
        input_shape = cls._get_input_shape(model_name)
        if model_name == MODEL_INCEPTION_V3:
            model = InceptionV3(input_shape=input_shape, include_top=False, weights="imagenet")
        elif model_name == MODEL_RESNET50:
            model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
        elif model_name == MODEL_INCEPTION_RESNET_V2:
            model = InceptionResNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
        elif model_name == MODEL_MOBILE:
            model = MobileNet(input_shape=input_shape, include_top=False, weights="imagenet")
        elif model_name == MODEL_VGG16:
            model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
        elif model_name == MODEL_VGG19:
            model = VGG19(input_shape=input_shape, include_top=False, weights="imagenet")
        elif model_name == MODEL_DENSE_NET_201:
            blcks = [6, 12, 48, 32]
            model = DenseNet(blocks=blcks, input_shape=input_shape, include_top=False, weights="imagenet")
        elif model_name == MODEL_NASNET_LARGE:
            model = NASNetLarge(input_shape=input_shape, include_top=False, weights="imagenet")
        elif model_name == MODEL_NASNET_MOBILE:
            model = NASNetMobile(input_shape=input_shape, include_top=False, weights="imagenet")
        else:
            raise ValueError("Error: no such model '{}'...".format(model_name))
        layer_by_name = {layer.name: layer for layer in model.layers}
        if layer != "last":
            model = Model(inputs=model.inputs, outputs=[layer_by_name[layer].output])
        if filters is not None:
            slices = list()
            for filter in filters:
                slices.append((lambda s: Lambda(
                    function=lambda t: tf.expand_dims(t[:, :, :, s], axis=3),
                    output_shape=list(model.output_shape[:3]) + [1]
                )(model.output))(filter))
            model = Model(inputs=model.inputs, outputs=[Concatenate(axis=-1)(slices)])
        if reduction == "average_pooling":
            glb_avg = GlobalAveragePooling2D()(model.output)
            model = Model(inputs=model.inputs, outputs=[glb_avg])
        if weights is not None and weights != "imagenet":
            model.load_weights(weights, by_name=True)
        return model

    @staticmethod
    def _get_preprocessing(model_name=None):
        if model_name == MODEL_INCEPTION_V3:
            return inceptionv3_preprocess
        elif model_name == MODEL_RESNET50:
            return resnet50_preprocess
        elif model_name == MODEL_INCEPTION_RESNET_V2:
            return inceptionresnetv2_preprocess
        elif model_name == MODEL_MOBILE:
            return mobilenet_preprocess
        elif model_name == MODEL_VGG16:
            return lambda x: vgg16_preprocess(x, mode="tf")
        elif model_name == MODEL_VGG19:
            return lambda x: vgg19_preprocess(x, mode="tf")
        elif model_name == MODEL_DENSE_NET_201:
            return densenet_preprocess
        elif model_name == MODEL_NASNET_LARGE or model_name == MODEL_NASNET_MOBILE:
            return nasnet_preprocess
        else:
            raise ValueError("Error: no such model '{}'...".format(model_name))

    @staticmethod
    def _get_input_shape(model_name=None):
        if model_name == MODEL_INCEPTION_V3:
            return 299, 299, 3
        elif model_name == MODEL_RESNET50:
            return 224, 224, 3
        elif model_name == MODEL_INCEPTION_RESNET_V2:
            return 299, 299, 3
        elif model_name == MODEL_MOBILE:
            return 224, 224, 3
        elif model_name == MODEL_VGG16:
            return 224, 224, 3
        elif model_name == MODEL_VGG19:
            return 224, 224, 3
        elif model_name == MODEL_DENSE_NET_201:
            return 224, 224, 3
        elif model_name == MODEL_NASNET_LARGE:
            return 331, 331, 3
        elif model_name == MODEL_NASNET_MOBILE:
            return 224, 224, 3
        else:
            raise ValueError("Error: no such model '{}'...".format(model_name))
