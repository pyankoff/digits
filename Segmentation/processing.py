import numpy as np
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt

def cut_line(image):
    h, l = image.shape
    ver = np.sum(image, axis=1)
    hor = np.sum(image, axis=0)
    ver_nonzero = np.flatnonzero(ver)
    #print ver
    hor_nonzero = np.flatnonzero(hor)
    ver_st = max(ver_nonzero[0]-3, 0)
    ver_fin = min(ver_nonzero[len(ver_nonzero)-1]+3, h)
    hor_st = max(hor_nonzero[0]-20, 0)
    hor_fin = min(hor_nonzero[len(hor_nonzero)-1]+20, l)
    #print h, l, ver_st, ver_fin, hor_st, hor_fin
    return 255 - image[ver_st:ver_fin, hor_st:hor_fin]


def clearing(raw_image):
    
    print "raw_image"
    plt.imshow(raw_image, cmap=plt.cm.gray)
    plt.show()

    print "background removed"
    #raw_image = ndimage.gaussian_filter(raw_image, 0.1)
    raw_image = np.where(raw_image < np.mean(raw_image) - 
                           5 * np.std(raw_image), 255, 0)
    #print raw_image
    plt.imshow(raw_image, cmap=plt.cm.gray)
    plt.show()


    print "denoise, fatten"
    raw_image = ndimage.median_filter(raw_image, 2, mode='reflect')
    raw_image = ndimage.gaussian_filter(raw_image, sigma=0.9, mode='reflect')
    raw_image = ndimage.binary_opening(raw_image, structure=np.ones((5,5)))
    plt.imshow(raw_image, cmap=plt.cm.gray)
    plt.show()
    #raw_image = np.where(raw_image < np.median(raw_image), 255, 0)

    print "cut"
    result = cut_line(raw_image)
    plt.imshow(result, cmap=plt.cm.gray)
    plt.show()
    return 255 - result
    

#raw_image = misc.imread('page.png', flatten=1)
#clearing()


