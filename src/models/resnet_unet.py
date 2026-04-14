"""
src/models/resnet_unet.py
ResNet50-backbone UNet for binary road segmentation.
Clean, modular, configurable via ModelConfig.
"""
from __future__ import annotations

import logging
from typing import List

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    UpSampling2D,
)
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)

# Skip-connection layer names from ResNet50
_SKIP_LAYER_MAP = {
    "s1": "conv1_relu",
    "s2": "conv2_block3_out",
    "s3": "conv3_block4_out",
    "s4": "conv4_block6_out",
    "bottleneck": "conv5_block3_out",
}


def _decoder_block(x: tf.Tensor, skip: tf.Tensor,filters: int, dropout: float, name: str,) -> tf.Tensor:
    
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", name=f"{name}_upsample")(x)
    x = Concatenate(name=f"{name}_concat")([x, skip])
    x = Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    x = BatchNormalization(name=f"{name}_bn2")(x)
    if dropout > 0:
        x = Dropout(dropout, name=f"{name}_drop")(x)
    return x


def build_resnet_unet(
    img_size: int,
    decoder_filters: List[int],
    dropout_rate: float,
    backbone_weights: str = "imagenet",
    freeze_encoder: bool = False,
) -> Model:
    """
    Build ResNet50-UNet.

    Args:
        img_size:        square input size (e.g. 256)
        decoder_filters: list of 4 filter counts [512, 256, 128, 64]
        dropout_rate:    dropout after each decoder block
        backbone_weights: 'imagenet' or None
        freeze_encoder:  whether to freeze ResNet50 weights

    Returns:
        Compiled-ready Keras Model
    """
    assert len(decoder_filters) == 4, "Need exactly 4 decoder filter counts"

    inputs = Input(shape=(img_size, img_size, 3), name="input_image")

    base = ResNet50(
        weights=backbone_weights,
        include_top=False,
        input_tensor=inputs,
    )
    base.trainable = not freeze_encoder

    # Encoder skip connections
    s1 = base.get_layer(_SKIP_LAYER_MAP["s1"]).output          # 128x128x64
    s2 = base.get_layer(_SKIP_LAYER_MAP["s2"]).output          # 64x64x256
    s3 = base.get_layer(_SKIP_LAYER_MAP["s3"]).output          # 32x32x512
    s4 = base.get_layer(_SKIP_LAYER_MAP["s4"]).output          # 16x16x1024
    b1 = base.get_layer(_SKIP_LAYER_MAP["bottleneck"]).output  # 8x8x2048

    # Decoder
    d1 = _decoder_block(b1, s4, decoder_filters[0], dropout_rate, "dec1")   # 16x16
    d2 = _decoder_block(d1, s3, decoder_filters[1], dropout_rate, "dec2")   # 32x32
    d3 = _decoder_block(d2, s2, decoder_filters[2], dropout_rate, "dec3")   # 64x64
    d4 = _decoder_block(d3, s1, decoder_filters[3], dropout_rate, "dec4")   # 128x128

    # Final upsample to original size
    d5 = UpSampling2D((2, 2), name="final_upsample")(d4)                    # 256x256
    outputs = Conv2D(1, 1, activation="sigmoid", name="road_mask")(d5)

    model = Model(inputs=base.input, outputs=outputs, name="resnet_unet")

    logger.info(
        "Built ResNet-UNet | params: %s | trainable: %s",
        f"{model.count_params():,}",
        f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}",
    )
    return model


def build_model_from_config(data_cfg, model_cfg) -> Model:
    """Convenience wrapper that unpacks config objects."""
    return build_resnet_unet(
        img_size=data_cfg.img_size,
        decoder_filters=model_cfg.decoder_filters,
        dropout_rate=model_cfg.dropout_rate,
        backbone_weights=model_cfg.backbone_weights,
        freeze_encoder=model_cfg.freeze_encoder,
    )
