import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dropout, GlobalAveragePooling2D, Input, ReLU, Reshape, UpSampling2D
from tensorflow.keras import Model
from config import WIDTH, HEIGHT, OUTPUT_CHANNELS, DROPOUT

class PSPNet:
    def __init__(self):
        self.use_tf_addons = False

    def build_model(self) -> Model:
        inputs = Input(shape=(HEIGHT, WIDTH, 3), name="inputs")
        with tf.name_scope('backbone'):
            x = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=(HEIGHT, WIDTH, 3))(inputs)
            x.trainable = False

        pyramid_pool0 = self._build_pyramid_pooling(x, 'pyramid_pooling0', tuple(x.get_shape().as_list()[1:-1]), 1)
        pyramid_pool1 = self._build_pyramid_pooling(x, 'pyramid_pooling1', tuple(x.get_shape().as_list()[1:-1]), 2, use_global_pool=False)
        pyramid_pool2 = self._build_pyramid_pooling(x, 'pyramid_pooling2', tuple(x.get_shape().as_list()[1:-1]), 4, use_global_pool=False)
        pyramid_pool3 = self._build_pyramid_pooling(x, 'pyramid_pooling3', tuple(x.get_shape().as_list()[1:-1]), 8, use_global_pool=False)

        concat = Concatenate(name='concat0')([x, pyramid_pool0, pyramid_pool1, pyramid_pool2, pyramid_pool3])

        with tf.name_scope('classifier'):
            conv0 = Conv2D(512, kernel_size=3, padding="same", kernel_initializer="he_normal", name='classifier_conv0')(concat)
            conv0 = BatchNormalization(name='batchnorm')(conv0)
            conv0 = ReLU(name='relu')(conv0)
            conv0 = Dropout(DROPOUT, name='dropout')(conv0)
            conv1x1 = Conv2D(OUTPUT_CHANNELS, 1, padding="same", name='classifier_conv1x1', kernel_initializer="he_normal", activation=None)(conv0)

        upsample = UpSampling2D((32, 32), interpolation='bilinear', name='upsample')(conv1x1)
        outputs = upsample

        return Model(inputs, outputs, name='PSPNet')


    def _build_pyramid_pooling(self, x, scope_name: str, output_size: tuple, bin: int, use_global_pool: bool = True, conv_filters: int = 512):
        with tf.name_scope(scope_name):
            if use_global_pool:
                x = GlobalAveragePooling2D(name='global_avg_pool2d')(x)
                x = Reshape((1, 1, x.get_shape().as_list()[-1]), name='reshape')(x)
            else:
                x = AveragePooling2D(pool_size=bin, name='avg_pool2d_{}'.format(scope_name))(x)

            x = Conv2D(filters=conv_filters, kernel_size=1, activation='relu', kernel_initializer="he_normal", name='1x1_conv_{}'.format(scope_name))(x)
            if bin > 1:
                x = UpSampling2D(size=bin, interpolation='bilinear', name='bilinear_upsample_{}'.format(scope_name))(x)
            else:
                x = UpSampling2D(size=output_size, interpolation='bilinear', name='bilinear_upsample_{}'.format(scope_name))(x)

            return x
