import cv2
import numpy as np
import random
import os,sys
from collections import Counter
import time

def group(res_arr,size):
    for path in os.listdir("images"):
        print(path)
        t = time.time()
        base("images/"+path,res_arr, size)
        print("Image time", time.time()-t)
    

def base(path,res_arr,size):
    #Load in the image
    t1 = time.time()
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h,w = im.shape[:2]
    hsv_range = [180,255,255]
    div = [hsv_range[i]/res_arr[i] for i in range(len(hsv_range))]
    im = np.array(im/div,dtype="uint8")
    #Shift neighbors
    wp,wm,hp,hm = shift(im)
    t2 = time.time()
    print("Set up time", t2-t1)
    #For each neighbor list convert to neighbor array
    seed_value = wp[0,0]
    neighbors = [] #[side][color type][color value]{key:color value: value: number}
    for cvt in [wp,wm,hp,hm]:
        neighbors.append(neighbor_array(im,cvt,res_arr))
    t3 = time.time()
    print("Neighbor time", t3-t2)
    out = grow_image_00(neighbors, size, res_arr,seed_value)
    out = np.array(out*div,dtype="uint8")
    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    t4 = time.time()
    print("Grow time", t4-t3)
    cv2.imwrite("proc"+path.split("images")[1],out)
    t5 = time.time()
    print("Write time", t5-t4)
    

#This is going to be really slow because I don't have a good solution for natural growth
def grow_image_00(neighbors, size,hsv_range, seed_value):
    w,h = size
    blank = np.zeros((h+2,w+2,len(hsv_range)),dtype="uint8")
    blank[0,:] = seed_value
    blank[:,0] = seed_value
    audit = []
    for clr in range(len(hsv_range)):
        for i in range(1,h+1):
            for j in range(1,w+1):
                wp = blank[i,j+1,clr]
                wm = blank[i,j-1,clr]
                hp = blank[i+1,j,clr]
                hm = blank[i-1,j,clr]
                target = np.zeros((5,hsv_range[clr]+1))
                best_hit = 0
                for x in range(hsv_range[clr]+1):
                    vals = np.array([neighbors[0][clr][wp][x],neighbors[1][clr][wm][x],neighbors[2][clr][hp][x],neighbors[3][clr][hm][x]])
                    count = np.count_nonzero(vals)
                    target[count,x]+=np.sum(vals)
                    if count>best_hit:
                        best_hit=count
                audit.append(best_hit)
                blank[i,j,clr]=random.choices(list(range(hsv_range[clr]+1)), weights=target[best_hit])[0]
    blank = blank[1:h+1,1:w+1]
    return blank
                

def neighbor_array(im,cvt,hsv_range):
    n_arr = [] #[color type][color value][n_value]=count
    for clr in range(len(hsv_range)):
        i_clr = im[:,:,clr]
        cvt_clr = cvt[:,:,clr]
        n_list = np.zeros((hsv_range[clr]+1, hsv_range[clr]+1))
        #This step is now less expensive
        for i in range(hsv_range[clr]+1):
            val, cnt = np.unique(cvt_clr[i_clr==i], return_counts=True)
            for x in range(len(val)):
                n_list[i][val]=cnt
        n_arr.append(n_list)
    return n_arr
        

def shift(im):
    wp = np.roll(im,100,axis=1)
    wm = np.roll(im,-100,axis=1)
    hp = np.roll(im,100,axis=0)
    hm = np.roll(im,-100,axis=0)
    return wp,wm,hp,hm


t = time.time()
group([9,12,12],(100,100)) #8 seconds per image
#group([36,50,50],(100,100)) #20 seconds per image
#group([9,12,12],(250,250)) # 22 seconds per image
#group([180,255,255],(50,50)) # 60 seconds per image
print("Total time",time.time()-t)
