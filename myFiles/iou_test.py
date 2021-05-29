from sklearn.metrics import jaccard_similarity_score
import numpy as np
a = np.ones([2,2]) * 255
b = np.array([[1,1],[0,1]]) * 255
print(np.logical_or(a,b))
print(jaccard_similarity_score(a,b))
