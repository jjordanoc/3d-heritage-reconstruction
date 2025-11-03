import os
import random
import shutil

def main():
    base = "/home/juan-prochazka/Desktop/Grafica-Proyecto/3d-heritage-reconstruction/pointcloud_opt_simp/data"
    source = base + "/images_preprocessed"
    files = os.listdir(source)
    subset1 = random.sample(files,50)
    subset2 = random.sample(files,50)
    subset3 = random.sample(files,50)
    os.mkdir(base + "/sample1")
    os.mkdir(base + "/sample2")
    os.mkdir(base + "/sample3")
    for i in subset1:
        shutil.copyfile(source+"/"+i,base+"/sample1"+"/"+i)
    for i in subset2:
        shutil.copyfile(source+"/"+i,base+"/sample2"+"/"+i)
    for i in subset3:
        shutil.copyfile(source+"/"+i,base+"/sample3"+"/"+i)

main()