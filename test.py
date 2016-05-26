'''
Created on Mar 18, 2016

@author: chris
'''

import cv2
import glob
import numpy as np

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# images = glob.glob('/home/chris/Documents/Demo/webcam calib/*.jpg')
# print images
# img = None
# i = 0
# for fname in images:
#     i += 1
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#  
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
#  
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#  
#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         imgpoints.append(corners2)
#  
#         # Draw and display the corners
# #         img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
# #         cv2.imwrite('/home/chris/Documents/Demo/chessboard/img%s.jpg' % i, img)
#  
# rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# print rms
# print mtx
# print dist

# rms = 0.794076829857
# mtx = np.array([[  2.53812521e+03,   0.00000000e+00,   1.56421880e+03],
#                 [  0.00000000e+00,   2.53624853e+03,   9.55122004e+02],
#                 [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
# dist = np.array([[  2.28084980e-01,  -4.19226608e+00,  -6.61757284e-04,  -1.35155845e-02,
#     2.98831926e+01]])

# Shity webcam
rms = 0.306142424317
mtx = np.array([[ 760.83008632,    0.,          347.40767905],
                [   0.,          761.4234222,   220.74970401],
                [   0.,            0.,            1.,        ]])
dist = np.array([[-0.01467067,  0.46229717, -0.00128649, -0.00408053, -1.12199521]])

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap.open()
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if found == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        frame = draw(frame,corners2,imgpts)
    else:
        pass
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


