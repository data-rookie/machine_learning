from sklearn.decomposition import PCA
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
# from scipy import misc
import scipy.misc

all_picture=[]
for root, dir, file in os.walk('.\data\\yalefaces'):
    for i in file:
        if ".gif" in i:
            all_picture.append(i)

all=[]
for i in range(len(all_picture)):
    img = Image.open(r'.\data\yalefaces\\'+all_picture[i])
    img = np.array(img)
    all.append(img)
all=np.array(all)
# all.reshape(all.shape[1],all.shape[2],all.shape[0])
all=all.reshape(all.shape[0],-1)
pca = PCA(n_components=20)
pca.fit(all)#只接受两个维度的
for i in range(len(all_picture)):
    x = pca.transform(all[i].reshape(1, -1))
    recon = np.dot(pca.components_.T, x.T)
    result=np.squeeze(recon) + pca.mean_#############################################
    # misc.imsave('.\data\output\\'+all_picture[i].replace('.gif','.jpg'), result)
    result=result.reshape(243,320)
    im = Image.fromarray(result)
    if im.mode == "F":
        im = im.convert('RGB')
    location='.\data\output\\'+all_picture[i].replace('.gif','.jpg')
    im.save(location)
print('a')


