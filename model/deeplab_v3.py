import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, concatenate, Conv2D, Dropout, GlobalAveragePooling2D, Input, MaxPool2D, ReLU, Reshape, UpSampling2D
from tensorflow.keras import Model
from config import WIDTH, HEIGHT, OUTPUT_CHANNELS, DROPOUT

class DeeplabV3:
    def __init__(self, aspp_filters: int = 256):
        self.aspp_filters = aspp_filters

    def build_model(self) -> Model:
        inputs = Input(shape=(HEIGHT, WIDTH, 3), name="inputs")

        with tf.name_scope('encoder'):
            with tf.name_scope('backbone'):
                x = self.__downsample_block(inputs, 64, name="down_sample_block0")
                # 2 - downsample
                decoder_skip = self.__downsample_block(x, 128, name="down_sample_block1")
                # 3 - downsample
                skip = self.__downsample_block(decoder_skip, 256, name="down_sample_block2")
                # 4 - downsample
                x = self.__downsample_block(skip, 512, name="down_sample_block3")
                # 5 - bottleneck
                bottleneck = self.__double_conv_block(x, 1024, names=("bottleneck_conv0", "bottleneck_conv1"))

            with tf.name_scope('aspp'):
                with tf.name_scope('conv1x1_pyramid'):
                    conv1x1 = Conv2D(self.aspp_filters, 1, padding='same', use_bias=False, name='conv1x1')(bottleneck)
                    conv1x1 = BatchNormalization(name='conv1x1_pyramid_bn')(conv1x1)
                    conv1x1 = ReLU(name='conv1x1_pyramid_relu')(conv1x1)

                with tf.name_scope('pyramid_6'):
                    conv_6 = Conv2D(self.aspp_filters, kernel_size=3, padding='same', dilation_rate=1, use_bias=False, name='conv_6')(bottleneck)
                    conv_6 = BatchNormalization(name='pyramid_6_bn')(conv_6)
                    conv_6 = ReLU(name='pyramid_6_relu')(conv_6)

                with tf.name_scope('pyramid_12'):
                    conv_12 = Conv2D(self.aspp_filters, kernel_size=3, padding='same', dilation_rate=1, use_bias=False, name='conv_12')(bottleneck)
                    conv_12 = BatchNormalization(name='pyramid_12_bn')(conv_12)
                    conv_12 = ReLU(name='pyramid_12_relu')(conv_12)

                with tf.name_scope('pyramid_18'):
                    conv_18 = Conv2D(self.aspp_filters, kernel_size=3, padding='same', dilation_rate=1, use_bias=False, name='conv_18')(bottleneck)
                    conv_18 = BatchNormalization(name='pyramid_18_bn')(conv_18)
                    conv_18 = ReLU(name='pyramid_18_relu')(conv_18)

                with tf.name_scope('aspp_pooling'):
                    pooling = AveragePooling2D(pool_size=(2, 2), name='aspp_avg_pool')(skip)
                    pooling = Conv2D(self.aspp_filters, 1, padding='same', use_bias=False, name='aspp_pooling_conv')(pooling)
                    pooling = BatchNormalization(name='aspp_pooling_bn')(pooling)
                    pooling = ReLU(name='aspp_pooling_relu')(pooling)

            concat = concatenate([conv1x1, conv_6, conv_12, conv_18, pooling], name='concat')
            encoder_output = Conv2D(256, 1, use_bias=False, name='encoder_output')(concat)

        with tf.name_scope('decoder'):
            decoder = Conv2D(256, 1, padding='same', use_bias=False, name='decoder_conv0')(decoder_skip)
            decoder = BatchNormalization(name='decoder_bn0')(decoder)
            decoder = ReLU(name='decoder_relu0')(decoder)

            encoder_output = UpSampling2D((4, 4), interpolation='bilinear', name='upsample_decoder')(encoder_output)

            decoder = concatenate([decoder, encoder_output], name='decoder_concat')

            with tf.name_scope('classifier'):
                decoder = Conv2D(256, 3, padding='same', use_bias=False, name='decoder_classifier_conv0')(decoder)
                decoder = BatchNormalization(name='decoder_classifier_bn0')(decoder)
                decoder = ReLU(name='decoder_classifier_relu0')(decoder)

                decoder = Conv2D(256, 3, padding='same', use_bias=False, name='decoder_classifier_conv1')(decoder)
                decoder = BatchNormalization(name='decoder_classifier_bn1')(decoder)
                decoder = ReLU(name='decoder_classifier_relu1')(decoder)

                decoder = Conv2D(OUTPUT_CHANNELS, 1, padding='same', activation='sigmoid', use_bias=False, name='decoder_conv1x1')(decoder)
                decoder = UpSampling2D((4, 4), interpolation='bilinear', name='decoder_output')(decoder)

        return Model(inputs, decoder, name='DeepLabv3')

    def __double_conv_block(self, inputs, n_filters: int, names: tuple):
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=names[0])(inputs)
        x = BatchNormalization()(x)
        concat = concatenate([x, inputs])
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=names[1])(concat)
        x = BatchNormalization()(x)
        x = concatenate([x, concat])
        return x

    def __downsample_block(self, x, n_filters: int, name: str):
        double_conv = self.__double_conv_block(x, n_filters, names=("{}_conv0".format(name), "{}_conv1".format(name)))
        pooled = MaxPool2D(2, name="{}_maxpool".format(name))(double_conv)
        pooled = Dropout(DROPOUT, name="{}_dropout".format(name))(pooled)
        return pooled
