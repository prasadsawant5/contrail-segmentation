import tensorflow as tf

@tf.keras.saving.register_keras_serializable(package="DiceLoss")
class DiceLoss(tf.keras.losses.Loss):

    def __init__(self, smooth: float = 1e-6, is_sigmoid: bool = False):
        super().__init__()
        self.smooth = smooth
        self.is_sigmoid = is_sigmoid

    def call(self, y_true, y_pred):
        if self.is_sigmoid:
            y_pred = tf.math.sigmoid(y_pred)
        # targets = K.flatten(y_true)
        # inputs = K.flatten(y_pred)

        numerator = (2 * tf.math.reduce_sum(y_true * y_pred)) + self.smooth
        denominator = tf.math.reduce_sum(y_pred + y_true) + self.smooth
        dice = numerator / denominator

        return 1 - dice
