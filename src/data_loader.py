import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, image_dir, mask_dir, config):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_height = config['img_height']
        self.img_width = config['img_width']
        self.normalization_factor = config['normalization_factor']
        self.mask_threshold = config['mask_threshold']
        self.test_size = config['test_size']
        self.random_state = config['random_state']
        
    def get_mask_name(self, img_name):
        prefix, rest = img_name.split("_", 1)
        return f"{prefix}_road_{rest}"
    
    def load_sample(self, img_name):
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, self.get_mask_name(img_name))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        road = mask[:,:,0] == self.mask_threshold
        mask = road.astype(np.float32)
        
        img = cv2.resize(img, (self.img_width, self.img_height))
        mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)
        
        return img, mask
    
    def load_data(self):
        images = os.listdir(self.image_dir)
        X, Y = [], []
        
        for img_name in images:
            img, mask = self.load_sample(img_name)
            X.append(img)
            Y.append(mask)
        
        X = np.array(X) / self.normalization_factor
        Y = np.array(Y).astype(np.float32)
        
        return X, Y
    
    def split_data(self, X, Y):
        return train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)