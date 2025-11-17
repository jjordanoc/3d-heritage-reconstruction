import random
import os
import math
import shutil

def find_keyframes(imgs_folder,dst_folder):
    files = os.listdir(imgs_folder)
    selection = random.sample(files,math.ceil(3*math.log2(len(files))))
    print(selection)
    for i in selection:
        shutil.copy2(imgs_folder + i,dst_folder+i)

find_keyframes("./data/clean_data/","./data/keyframes/random/")