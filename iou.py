from sklearn.metrics import jaccard_score
import numpy as np
import cv2
import glob
from mrcnn import *

a = np.random.randint(0,2,[30,30])
b = np.random.randint(0,2,[30,30])
print(a,b)

a1 = [1,1,0,1,0,0,0]
b1 = [1,1,1,0,0,0,0]

print(jaccard_score(a1,b1))

#print(a '\n' b)
print(jaccard_score(a.flatten(),b.flatten()))

