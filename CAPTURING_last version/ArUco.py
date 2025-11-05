import cv2, numpy as np
aruco=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
img=cv2.aruco.generateImageMarker(aruco, 0, 600)  # id=0, 600px
cv2.imwrite("aruco_30mm_id0.png", img)