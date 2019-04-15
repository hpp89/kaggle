import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd

path = "./inputs/cancerDetection/"
x_train_path = "train"
y_train_path = "train_labels.csv"

# img = cv2.imread()
y_train = pd.read_csv(os.path.join(path, y_train_path))
print(y_train.shape, y_train.head())

print(y_train.items())
train_dict = {y_train.iloc[idx].id: y_train.iloc[idx].label for idx in range(len(y_train))}
print(train_dict.)
# y_train_names = y_train["id"].values
#
# for name in tqdm(y_train_names, total=len(y_train_names)):
#     name = "{}train/{}.tif".format(path, name)
