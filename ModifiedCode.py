import re
import os
import json
import random
import time
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from SoccerNet.Downloader import getListGames, SoccerNetDownloader
from SoccerNet.utils import getListGames

start = time.time()

# Get list of training games
list_game_train = getListGames("train")
list_of_games_index = [getListGames("train").index(i) for i in list_game_train]

# Read all games from a specific file with annotations in frames
labels = open("/Users/mohannadhosam/Downloads/print_labels_copy.txt").readlines()

# Append all games with a specific space
list_game = []
for line in labels:
    for i in list_of_games_index:
        if f"g\t{i+1}\t" in line:
            new_line = re.sub(r"[\n\t]", "  ", line)
            list_game.append(new_line)

# Search for a specific event in the list and append the line to "list_frame" 
list_frame = [line for line in list_game if "  yellow  " in line]

# Features of 1st half and second half
features_half_1 = [np.load(os.path.join("/Users/mohannadhosam/Downloads/BachelorThesisProject/PATH_DATASET/",game,"1_ResNET_TF2_PCA512.npy")) for game in list_game_train]
features_half_2 = [np.load(os.path.join("/Users/mohannadhosam/Downloads/BachelorThesisProject/PATH_DATASET/",game,"2_ResNET_TF2_PCA512.npy")) for game in list_game_train]

# Extract modified features from selected frames
features_modified_after = []

for line in list_frame[0:200]:
    split_line = line.split()
    if('h' + '  ' + '1'  in line):
        start_index = int(split_line[7])
        features_modified_after.append(features_half_1[int(split_line[1])][start_index:start_index+10])
    elif('h' + '  ' + '2'  in line):
        start_index = int(split_line[7])
        features_modified_after.append(features_half_2[int(split_line[1])][start_index:start_index+10])
        

# Prepare features for clustering
array = np.array(features_modified_after).reshape(200,-1,512)
label = []
scaling = StandardScaler()
for line in array:
    #print(line.shape)
    scaled_value = scaling.fit_transform(line)
    kmeans = KMeans(n_clusters=2,random_state=42).fit(scaled_value)
    label.append(kmeans.labels_)

# Write labels to file
with open("/Users/mohannadhosam/Documents/BachelorThesisProject/results_10Frames_after_yellowcard.txt", "w") as f:
    for l in label:
        f.write("Game=" + str(l.tolist()) + "\n")

# Load labels from file and clean up
with open("/Users/mohannadhosam/Documents/BachelorThesisProject/results_10Frames_after_yellowcard.txt") as f:
    list_20 = [line[line.index('[', 0, len(line)):].replace("\n", "") for line in f if 'Game' in line]

list_final = [line.replace(" ", "") for line in list_20]

counters = []
for line in list_final:
    count_zero = line.count("0")
    count_one = line.count("1")
    counters.extend([f"{count_zero}\t", f"{count_one}\t", "\n"])

with open("/Users/mohannadhosam/Documents/BachelorThesisProject/count_no_frames_after_new_yellowcard.txt", 'a') as f:
    f.writelines(counters)


file_path = "/Users/mohannadhosam/Documents/BachelorThesisProject/count_no_frames_after_new_yellowcard.txt"

with open(file_path) as f:
    data = [list(map(int, line.strip().split())) for line in f]

max_numbers = [max(line) for line in data]

plt.hist(max_numbers, bins=30, edgecolor='black')
plt.show()

end = time.time()

print(end - start)
