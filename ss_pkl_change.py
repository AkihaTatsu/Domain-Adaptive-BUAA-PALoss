import pickle
import cv2
import os
import numpy as np
from tqdm import tqdm

# speed-up using multithreads
cv2.setUseOptimized(True)
cv2.setNumThreads(8)

path = input("Input pickle file path:")
with open(path, "rb") as f:
    data = pickle.load(f)

img_path = input("Input image folder path:")
walk_res = os.walk(img_path)
img_infos = []
for root, dirs, files in walk_res:
    for f in files:
        img_infos.append([f, os.path.join(root, f)])

output_path = input("Input output pickle file path:")

print("Operation type: 1 for offset -> vertex, 2 for vertex -> offset")
command = input("Input your command:")

for i in tqdm(range(len(data['indexes'])), ascii=True):
    index = data['indexes'][i]
    for img_info in img_infos:
        if index == int(img_info[0].split(".")[0]):
            im = cv2.imread(img_info[1]) 
            rects = data['boxes'][i]

            if command == '1':
                new_rects = []
                for rect in rects:
                    new_rect = np.array([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
                    # check if valid
                    if new_rect[2] > im.shape[1] or new_rect[3] > im.shape[0]:                    
                        raise ValueError(f"This file is not offset. Pic {img_info[1]}, original rect is {rect}, width of pic is {im.shape[1]}, height of pic is {im.shape[0]}")
                    new_rects.append(new_rect)

            if command == '2':
                new_rects = []
                for rect in rects:
                    new_rect = np.array([rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]])
                    # check if valid
                    if new_rect[2] <= 0 or new_rect[3] <= 0:
                        raise ValueError(f"This file is not vertex. Pic {img_info[1]}, original rect is {rect}")
                    new_rects.append(new_rect)
                    
            data['boxes'][i] = np.array(new_rects)

output_dir_parent_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir_parent_dir):
    os.makedirs(output_dir_parent_dir)
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
