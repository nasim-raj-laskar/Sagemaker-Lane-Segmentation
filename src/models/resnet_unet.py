"""src/models/resnet_unet.py — ResNet50-UNet architecture factory."""

from __future__ import annotations

import tensorflow as tf
from omegaconf import DictConfig
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D
from tensorflow.keras.models import Model

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_resnet_unet(cfg: DictConfig) -> Model:
    """Build a ResNet50 encoder + UNet decoder binary segmentation model.

    Architecture is fully driven by configs/model/resnet_unet.yaml.
    """
    mc = cfg.model
    bc = mc.backbone
    dc = mc.decoder

    # ── Encoder (frozen ResNet50 backbone) ──────────────────────────────────
    base = ResNet50(
        weights=bc.weights,
        include_top=False,
        input_shape=tuple(bc.input_shape),
    )
    base.trainable = bc.trainable
    logger.info("Backbone trainable=%s", bc.trainable)

    el = mc.encoder_layers
    s1 = base.get_layer(el.s1).output
    s2 = base.get_layer(el.s2).output
    s3 = base.get_layer(el.s3).output
    s4 = base.get_layer(el.s4).output
    bn = base.get_layer(el.bottleneck).output

    # ── Decoder ─────────────────────────────────────────────────────────────
    filters = dc.filters
    skips = [s4, s3, s2, s1]

    x = bn
    for f, skip in zip(filters, skips):
        x = UpSampling2D((mc.decoder.upsample_factor, mc.decoder.upsample_factor))(x)
        x = Concatenate()([x, skip])
        x = Conv2D(f, dc.kernel_size, padding=dc.padding, activation=dc.activation)(x)

    # Final upsample to match input resolution
    x = UpSampling2D((mc.decoder.upsample_factor, mc.decoder.upsample_factor))(x)

    # ── Head ────────────────────────────────────────────────────────────────
    outputs = Conv2D(
        mc.head.filters,
        mc.head.kernel_size,
        activation=mc.head.activation,
        name="segmentation_output",
    )(x)

    model = Model(base.input, outputs, name="resnet_unet")
    logger.info(
        "Model built — params: %s  trainable: %s",
        f"{model.count_params():,}",
        f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}",
    )
    return model


def unfreeze_encoder(model: Model, cfg: DictConfig) -> Model:
    """Unfreeze encoder layers for fine-tuning phase."""
    ft = cfg.model.fine_tuning
    unfreeze_from = ft.unfreeze_from_layer
    found = False
    for layer in model.layers:
        if layer.name == unfreeze_from:
            found = True
        if found:
            layer.trainable = True
    if not found:
        logger.warning("Layer '%s' not found — unfreezing entire model", unfreeze_from)
        model.trainable = True
    trainable_count = sum(tf.size(v).numpy() for v in model.trainable_variables)
    logger.info("Encoder unfrozen from '%s' — trainable params: %s", unfreeze_from, f"{trainable_count:,}")
    return model
