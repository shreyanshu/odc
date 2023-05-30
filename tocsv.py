# # from matplotlib.pyplot import imread
from PIL import Image
import numpy as np
import pandas as pd
import os
#
root = 'dataset_preprocessed_train\\'
# go through each directory in the root folder given above
for directory, subdirectories, files in os.walk(root):
    print(directory, subdirectories, files)
    for file in files:
        im = np.asanyarray(Image.open(os.path.join(directory, file)))
        value = im.flatten()
        value = np.hstack((directory[-1], value)) #1 #1024
        df = pd.DataFrame(value).T
        tf = df.sample(frac=1)
        with open('train.csv', 'a') as dataset:
            tf.to_csv(dataset, header=False, index=False)

df = pd.read_csv('train.csv')  # for train: train_initial.csv
df = df.sample(frac=1)
df.to_csv('train.csv', header=False, index=False)  # for train: train.csv