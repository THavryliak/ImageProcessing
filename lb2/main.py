import cv2
import matplotlib.pyplot as plt
from PIL import Image

brisk = cv2.BRISK_create()


def hamming(x, y):
    assert x.shape == y.shape

    return sum(el1 != el2 for el1, el2 in zip(x, y))


def cvBF(image1, image2, descriptors1, descriptors2, keypoints1, keypoints2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(match_img), plt.show()


def ownMatch(image1, image2, descriptors1, descriptors2, keypoints1, keypoints2):
    matches = []
    for i, el1 in enumerate(descriptors1):
        for j, el2 in enumerate(descriptors2):
            matches.append(cv2.DMatch(_distance=int(hamming(el1, el2)), _imgIdx=0, _queryIdx=i, _trainIdx=j))

    matches = sorted(matches, key=lambda x: x.distance)

    match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(match_img), plt.show()


img1 = cv2.imread('bird_part.jpg')
img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
img2 = cv2.imread('bird.jpg')

kp1, des1 = brisk.detectAndCompute(img1, None)
kp2, des2 = brisk.detectAndCompute(img2, None)

ownMatch(img1, img2, des1, des2, kp1, kp2)
cvBF(img1, img2, des1, des2, kp1, kp2)
