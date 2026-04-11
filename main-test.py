from src.data.preprocessing import img_to_mask_name
import cv2
import matplotlib.pyplot as plt

img_path = 'dataset/image/um_000001.png'

mask_path = img_to_mask_name(img_path)
print("Mask path:", mask_path)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print("Mask not found")
    exit()

plt.imshow(mask, cmap='gray')
plt.axis("off")
plt.savefig("mask_output.png")

print("Saved mask_output.png")