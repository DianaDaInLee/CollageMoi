#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import sys
import re
import numpy as np
from PIL import Image
from imageio import imread


# In[ ]:


def split_image(image_path, rows, cols, should_cleanup, should_quiet=False, output_dir=None):
    im = Image.open(image_path)
    im_width, im_height = im.size
    row_width = int(im_width / cols)
    row_height = int(im_height / rows)
    name, ext = os.path.splitext(image_path)
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = "./"
    for i in range(0, rows):
        for j in range(0, cols):
            box = (j * row_width, i * row_height, j * row_width +
                   row_width, i * row_height + row_height)
            outp = im.crop(box)
            outp_path = "split_" + str(i) + "_" + str(j) + ext
            outp_path = os.path.join(output_dir, outp_path)
            if not should_quiet:
                print("Exporting image tile: " + outp_path)
            outp.save(outp_path)
    if should_cleanup:
        if not should_quiet:
            print("Cleaning up: " + image_path)
        os.remove(image_path)


# In[ ]:


def CollageMoi(fldr_main, # Name of the folder where the main photo is located
               fldr_cand, # Name of the folder where all training photos are located
               img_main,  # Filename of the main photo
               img_final, # Filename of the final collage photo you want to save as
               nsplit_row, # INT: Number of rows of the main photo you want to split by
               nsplit_col, # INT: Number of columns of the main photo you want to split by
               space = 0, # INT: White space between individual tiles in a main collage 
               dist_method = cv2.HISTCMP_INTERSECT): # Distance metric in comparing main photo against training photos. Choose from # HISTCMP_CORREL, HISTCMP_CHISQR, HISTCMP_INTERSECT, HISTCMP_BHATTACHARYYA, HISTCMP_HELLINGER, HISTCMP_CHISQR_ALT, HISTCMP_KL_DIV

    print('#1 Slicing and vectoring main images\n')
    # 1) Slice image
    name, ext = img_main.split('.')
    im = cv2.imread(os.path.join(fldr_main, img_main))
    im = cv2.resize(im, (nsplit_row, nsplit_col), interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(fldr_main + '/' + name + '_resized.' + ext, im)
    im = cv2.imread(fldr_main + '/' + name + '_resized.' + ext)
    split_image(fldr_main + '/' + name + '_resized.' + ext, rows = nsplit_row, cols = nsplit_col, 
                should_cleanup = False, should_quiet = True, output_dir = fldr_main)

    # 2) Vectorize the color scheme
    col_main = {}
    for tile in os.listdir(fldr_main):
        if len(re.findall('split', tile))>0:
            img = cv2.imread(os.path.join(fldr_main, tile))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            col_main[tile] = hist

    print('#2 Vectorizing candidate images\n')
    # 1) Identify the size of a main image
    h, w, _ = cv2.imread(fldr_main + '/' + os.listdir(fldr_main)[0]).shape
    
    col_cand = {}
    for filename in os.listdir(fldr_cand):
        if len(re.findall('jpeg|jpg|png', filename))>0:
            # 2) Import candidate image and resize to be same as the main image
            img = cv2.imread(os.path.join(fldr_cand, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w*200, h*200))
            # plt.imshow(img, cmap = plt.cm.Spectral)
        # 3) Vectorize the color scheme
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        col_cand[filename] = hist
    del img
    del hist
    
    print('#3 Identifying the best matches\n')
    best_match = {}
    for main in col_main:
        distance = {}
        for cand in col_cand:
            distance[cand] = cv2.compareHist(col_main[main], col_cand[cand], dist_method)
        best_match[main] = max(distance, key = distance.get)
    del col_cand
    del col_main
    
    print('#3 Creating a collage\n')
    image_dict = {}
    # 1) Loop through the files in the directory
    for best in best_match:
        position = re.findall('_[0-9]+_[0-9]+', best)[0]
        _, row, col = position.split("_")
        row, col = int(row), int(col)
        img = Image.open(os.path.join(fldr_cand, best_match[best])).resize((w*200, h*200))
        image_dict[(row, col)] = img
    del best_match

    # 2) Calculate the dimensions of the final image
    max_row = max(row for row, _ in image_dict.keys())
    max_col = max(col for _, col in image_dict.keys())
    w, h = image_dict[(1, 1)].size

    final_width = (max_col + 1) * w
    final_height = (max_row + 1) * h

    background_width = final_width + (space*max_col)-space
    background_height = final_height + (space*max_row)-space

    # 3) Create an empty image to hold the final grid
    final_image = Image.new("RGB", (background_width, background_height))
    del final_width
    del final_height
    del background_width
    del background_height

    # 4) Populate the final image with individual images
    for row in range(max_row + 1):
        for col in range(max_col + 1):
            image = image_dict.get((row, col))
            if image:
                final_image.paste(image, (col * w, row * h))

    # 5) Save or display the final image
    final_image.save(img_final)
#     final_image.show()
    
    # 6) Removing split images
    for filename in os.listdir(fldr_main):
        if len(re.findall('split', filename))>0:
            os.remove(os.path.join(fldr_main, filename))

