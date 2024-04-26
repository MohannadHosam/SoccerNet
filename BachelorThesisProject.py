#libraries used
import re
from SoccerNet.Downloader import SoccerNetDownloader
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from SoccerNet.utils import getListGames
import os
import time
from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader

start = time.time()

###################################################################################################
#list_game_train is the list containing all training games which are 300 matches
list_game_train = getListGames("train")
#getting indices of all 300 matches to be used in list_game when accessing matches
list_of_games_index = []
for i in list_game_train:
  list_of_games_index.append(getListGames("train").index(i))

#Reading all games from a specific file with annotations in frames
labels = open("/Users/mohannadhosam/Downloads/print_labels_copy.txt").readlines()

#Appending all games with a specific space
list_game = []
for line in labels:
    for i in list_of_games_index:
        if 'g' +'\t'+ str(i + 1) +'\t' in line: #if we remove the last \t it will get all instances of 1 ex. 14 10 100
            newLine = re.sub(r"[\n\t]", "  ", line) #if we find a match replace the tab and newline with spaces
            list_game.append(newLine) #append this line to an array to later use

#search for a specific event in the list and append the line to "list_frame" 
list_frame = []
for line in list_game:
   if '  ' + 'yellow' + '  ' in line:
       list_frame.append(line)

#features of 1st half and second half
features_half_1 = []
features_half_2 = []
for game in list_game_train:
    features_half_1.append(np.load(os.path.join("/Users/mohannadhosam/Downloads/BachelorThesisProject/PATH_DATASET/",game,"1_ResNET_TF2_PCA512.npy")))

    features_half_2.append(np.load(os.path.join("/Users/mohannadhosam/Downloads/BachelorThesisProject/PATH_DATASET/",game,"2_ResNET_TF2_PCA512.npy")))


features_modified = []
features_modified_before = []
features_modified_after= []
for line in list_frame[0:200]:
    if('h' + '  ' + '1'  in line):
        features_modified_after.append(features_half_1[int(line.split()[1])][int(line.split()[7]):int(line.split()[7])+10])
        #print(line)
    elif('h' + '  ' + '2'  in line):
        #print(line)
        features_modified_after.append(features_half_2[int(line.split()[1])][int(line.split()[7]):int(line.split()[7])+10])
         

# file = open("/Users/mohannadhosam/Desktop/BachelorResultsSoFar/results_yellow_30Frames.txt",'a')
# for line in features_modified:
#     for i in line:
#         file.write(str(i))

 
#features_modified = np.array([features_modified],dtype=object).reshape(200,1)
#features_modified_np = np.array([features_modified_list],dtype=object).reshape(1,-1)

label = []
scaling = StandardScaler()
for line in features_modified_after:
    scaled_value = scaling.fit_transform(line)
    print(scaled_value)
    kmeans = KMeans(n_clusters=2,random_state=1).fit(scaled_value)
    label.append(kmeans.labels_)
#print(label)
##################################################################################
###########
# for i in label:
#      s = np.array_str(i)
#      file = open("/Users/mohannadhosam/Desktop/BachelorResultsSoFar/results_10Frames_after_yellow.txt",'a')
#      file.write("Game" +"=" + s + "\n")
#      file.close()
#########
file = open("/Users/mohannadhosam/Desktop/BachelorResultsSoFar/results_10Frames_after_yellow.txt").readlines()
#list_range = []
#for i in range(50):
# list_range.append(i)
list_30= []
for line in file:
   if 'Game'in line:
       list_30.append(line[line.index('[',0,len(line)):])
 
list_without_space = []  
for line in list_30:
   new_line = re.sub(r"[\n\t]","",line)
   list_without_space.append(new_line)
 
list_final = []
for line in list_without_space:
   new_Line = re.sub(r" ","",line)
   list_final.append(new_Line)

count_zero= 0
count_one = 0
count_two = 0
counter_array = []
for line in list_final:
   for i in range(1,9):
       if(line[i+1] == '0'):
            count_zero+=1
       if(line[i+1] == '1'):
            count_one+=1
    #    if(line[i]=='2'):
    #        count_two+=1
   counter_array.append(str(count_zero) + "\t") 
   counter_array.append(str(count_one) +'\t')
#    counter_array.append(str(count_two) + '\t')
   counter_array.append('\n')
   count_zero = 0
   count_one = 0 
#    count_two = 0
      
      
for i in counter_array:      
    file = open("/Users/mohannadhosam/Desktop/BachelorResultsSoFar/count_no_frames_after_new_yellow.txt",'a')
    file.write(i)
    file.close()

#plt.hist(counter_array,bins = 30,edgecolor='black')
#plt.show()

#print(count_zero)
#print(count_one)
#print(count_two) 

#print(counter_array)
           #after_array.append(line[15+i+1])
           
#        else:
#            break
#    count_array_after.append(count)
#    count = 0
 
# before_array = []
# for line in list_final:
#    for i in range(30):
#        if(line[15-i-1] == line[15]):
#            before_array.append(line[15-i-1])
#            count+=1
#        else:
#            break
#    count_array_before.append(count)
#    count = 0
#print(count_array_before)
# print("--------------------------")
#print(count_array_after)
#f1 = plt.figure(1)
#plt.hist(count_array_before,bins = 30,edgecolor='black')
#f2 = plt.figure(2)
#plt.hist(count_array_after,bins = 30,edgecolor='black')
#plt.show()
##################################################################################################
 
file = open("/Users/mohannadhosam/Desktop/BachelorResultsSoFar/count_no_frames_after_new_yellow.txt").readlines()
max_numbers = []
for line in file:
    numbers = [int(x) for x in line.strip().split()]
    max_numbers.append(max(numbers))

#print(max_numbers)

plt.hist(max_numbers,bins = 30,edgecolor='black')
plt.show()

end = time.time()

print(end - start)