import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Conv2DTranspose, concatenate, Dropout, Input, MaxPool2D
from tensorflow.keras import Model
from config import WIDTH, HEIGHT, OUTPUT_CHANNELS


class UNet:
    def __init__(self, dropout: float):
        self.dropout = dropout

    def build_model(self) -> Model:
        inputs = Input(shape=(HEIGHT, WIDTH, 3), name="inputs")
        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = self.__downsample_block(inputs, 64, name="down_sample_block0")
        # 2 - downsample
        f2, p2 = self.__downsample_block(p1, 128, name="down_sample_block1")
        # 3 - downsample
        f3, p3 = self.__downsample_block(p2, 256, name="down_sample_block2")
        # 4 - downsample
        f4, p4 = self.__downsample_block(p3, 512, name="down_sample_block3")
        # 5 - bottleneck
        bottleneck = self.__double_conv_block(p4, 1024, names=("bottleneck_conv0", "bottleneck_conv1"))
        # decoder: expanding path - upsample
        # 6 - upsample
        u6 = self.__upsample_block(bottleneck, f4, 512, name="upsample_block0")
        # 7 - upsample
        u7 = self.__upsample_block(u6, f3, 256, name="upsample_block1")
        # 8 - upsample
        u8 = self.__upsample_block(u7, f2, 128, name="upsample_block2")
        # 9 - upsample
        u9 = self.__upsample_block(u8, f1, 64, name="upsample_block3")

        outputs = Conv2D(OUTPUT_CHANNELS, 1, padding="same", activation="sigmoid", kernel_initializer="he_normal", name="outputs")(u9)

        return Model(inputs, outputs, name="UNet")

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

    def __downsample_block(self, x, n_filters: int, name: str) -> tuple:
        f = self.__double_conv_block(x, n_filters, names=("{}_conv0".format(name), "{}_conv1".format(name)))
        p = MaxPool2D(2, name="{}_maxpool".format(name))(f)
        p = Dropout(self.dropout, name="{}_dropout".format(name))(p)
        return f, p

    def __upsample_block(self, x, conv_features, n_filters: int, name: str):
        # upsample
        x = Conv2DTranspose(n_filters, 3, 2, padding="same", name="{}_conv_transpose".format(name))(x)
        # concatenate
        x = concatenate([x, conv_features])
        # dropout
        x = Dropout(self.dropout, name="{}_dropout".format(name))(x)
        # Conv2D twice with ReLU activation
        x = self.__double_conv_block(x, n_filters, names=("{}_conv0".format(name), "{}_conv1".format(name)))
        return x