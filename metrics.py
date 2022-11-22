import cv2
import numpy as np
from sklearn import metrics


p = cv2.imread("../DHF1K/val_images/601/saliency/0001.png")

y = cv2.imread("../DHF1K/annotation/0601/fixation/0001.png")


#resize image
y = y[:,0:512]
# cv2.imshow( "prediction", p)

# cv2.imshow( "ground truth", y)
print(p.shape)
print(y.shape)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Cross Validation Classification ROC AUC


