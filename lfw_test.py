from matplotlib import testing
from torchvision.datasets import LFWPairs

from deepface import DeepFace
import cv2
import numpy as np

if __name__ == '__main__':
    lfw_path = "datasets"
    lfw = LFWPairs(root=lfw_path,split="test",download=True)
    img1,img2,label = lfw.__getitem__(0)
    img1 = np.array(img1)
    img2 = np.array(img2)
    result = DeepFace.verify(img1_path = img1, img2_path = img2)
    print(result)
    print(label)
    # cv2.imshow("test",testImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()