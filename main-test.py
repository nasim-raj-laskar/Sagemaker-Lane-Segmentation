from src.models.resnet_unet import build_resnet_unet
model = build_resnet_unet(256, [512,256,128,64], 0.3, backbone_weights=None)
model.summary()