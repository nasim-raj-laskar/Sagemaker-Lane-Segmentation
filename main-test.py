from src.data.s3_io import read_image_from_s3, read_mask_from_s3
from src.data.preprocessing import img_to_mask_name
from src.data.s3_io import list_keys

keys = list_keys("self-driving-perceptron", "raw-data/image")
print(f"Found {len(keys)} images")
print(keys[:3])
img_key  = keys[0]                          # use the key from Test 2
mask_key = "raw-data/mask/" + img_to_mask_name(img_key.split("/")[-1])

img  = read_image_from_s3("self-driving-perceptron", img_key)
mask = read_mask_from_s3("self-driving-perceptron", mask_key)

print("Image shape:", img.shape)    # (375, 1242, 3)
print("Mask shape:",  mask.shape)   # (375, 1242)
print("Mask unique:", set(mask.flatten()[:1000].tolist()))  # {0.0, 1.0}