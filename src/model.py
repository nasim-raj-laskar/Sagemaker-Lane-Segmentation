import tensorflow as tf
from keras_unet_collection import models

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)

def create_unet_model(config):
    input_shape = (config['img_height'], config['img_width'], 3)
    model = models.unet_2d(
        input_shape,
        filter_num=[64, 128, 256, 512],
        n_labels=1,
        activation='ReLU',
        output_activation='Sigmoid'
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
        loss=dice_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.MeanIoU(num_classes=2)
        ]
    )
    
    return model