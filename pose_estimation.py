import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import glob

def timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print "Time [%s]: %f (seconds)" % (func.__name__, end-start)
    return wrapper

def orb_bf(img1, img2, ret_img=False):
    # Initialize the ORB detector
    orb = cv2.ORB_create()
    
    # Find the keypoints and descriptors for the 2 images
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)
    
    # Create the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match the 2 descriptors
    matches = bf.match(desc1, desc2)
    
    # Sort the matches by their distance
    matches = sorted(matches, key=lambda m:m.distance)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches[:10] ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches[:10] ]).reshape(-1,1,2)
    rt = cv2.estimateRigidTransform(src_pts, dst_pts, True)
    img3 = None
    
    if ret_img:
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], flags=2, outImg=None)
        
    return rt, img3
        

def ss_flann(img1, img2, algo='SIFT', ret_img=False):
    MIN_MATCH_COUNT = 10
    
    if algo == 'SIFT':
        alg = cv2.xfeatures2d.SIFT_create()
    elif algo == 'SURF':
        alg = cv2.xfeatures2d.SURF_create() 
    
    kp1, desc1 = alg.detectAndCompute(img1, None)
    kp2, desc2 = alg.detectAndCompute(img2, None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1,desc2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
            
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    img3 = None
    if len(good)>MIN_MATCH_COUNT and ret_img:
            
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w,_ = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    
    rt = cv2.estimateRigidTransform(src_pts, dst_pts, True)
    return rt, img3

def orb_flann(img1, img2, ret_img=False):
    orb = cv2.ORB_create()
    
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)
    
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    if ret_img:
        matchesMask = [[0,0] for i in xrange(len(matches))]
        
    good = []
    for i, (m,n) in enumerate(matches):
        if m.distance < .70*n.distance:
            if ret_img:
                matchesMask[i] = [1,0]
            good.append(m)
            
            
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    rt = cv2.estimateRigidTransform(src_pts, dst_pts, True)
    
    img3 = None
    if ret_img:
        draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    
    return rt, img3

def ss_bf(img1, img2, algo='SIFT', ret_img=False):
    if algo == 'SIFT':
        alg = cv2.xfeatures2d.SIFT_create()
    elif algo == 'SURF':
        alg = cv2.xfeatures2d.SURF_create()
        
    # find the keypoints and descriptors with SURF
    kp1, des1 = alg.detectAndCompute(img1,None)
    kp2, des2 = alg.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img3 = None
    if ret_img:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
    rt = cv2.estimateRigidTransform(src_pts, dst_pts, True)
    return rt, img3
#     if viz:
#         plt.subplot(331), plt.imshow(img1), plt.title('Original Image (img1)'), plt.xticks([]), plt.yticks([])
#         plt.subplot(332), plt.imshow(img2), plt.title('Transformed Image (img2)'), plt.xticks([]), plt.yticks([])
#         plt.subplot(333), plt.imshow(img3), plt.title("'img1' Descriptors vs. 'img2'"), plt.xticks([]), plt.yticks([])
#         
#         rows, cols, _ = img1.shape
#         rotated_img = cv2.warpAffine(img2, cv2.invertAffineTransform(rt), (cols, rows))
#         plt.subplot(334), plt.imshow(rotated_img), plt.xticks([]), plt.yticks([])
#         plt.tight_layout(pad=0.4, h_pad=0.1)
#         plt.show()
        
    
def build_pretty_thing(img1, img2):
    sift_bf_rt, sift_bf_img3 = ss_bf(img1, img2, 'SIFT', True)
    surf_bf_rt, surf_bf_img3 = ss_bf(img1, img2, 'SURF')
    rows, cols, _ = img1.shape
    
    # SIFT BF 
    plt.subplot(231), plt.imshow(img1), plt.title('Original Image (img1)'), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(img2), plt.title('Transformed Image (img2)'), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(sift_bf_img3), plt.title("'img1' Descriptors vs. 'img2'"), plt.xticks([]), plt.yticks([])
    # SIFT BF rebuild
    plt.subplot(234), plt.imshow(img1), plt.title('Original Image (img1)'), plt.xticks([]), plt.yticks([])
    sift_bf_inverse_transform = cv2.warpAffine(img2, cv2.invertAffineTransform(sift_bf_rt), (cols, rows))
    plt.subplot(235), plt.imshow(sift_bf_inverse_transform), plt.title("Inverse Affine (IA) of 'img2'"), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(ss_bf(img1, sift_bf_inverse_transform, 'SIFT', True)[1]), plt.title("'img1' Descriptors vs. IA of 'img2'"), plt.xticks([]), plt.yticks([])
    plt.suptitle("SIFT Algorithm with Brute Force Matcher", fontsize=18)
    plt.subplots_adjust(top=0.9)
         
    rows, cols, _ = img1.shape
#     rotated_img = cv2.warpAffine(img2, cv2.invertAffineTransform(rt), (cols, rows))
#     plt.subplot(334), plt.imshow(rotated_img), plt.xticks([]), plt.yticks([])
#     plt.tight_layout(pad=0.4, h_pad=1.3)
    plt.show()
    
def chessboard():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:,2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('/home/chris/Documents/Demo/*.jpg')
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Fine the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
            imgpoints.append(corners)
            
            # Draw and display corners
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    
def reconstruct_realtime(func, *args):
#     rigid_transform, img3 = func(**kwargs)
    img1 = None
    print "Press 'k' to take picture"
    cap = cv2.VideoCapture(1)
    # While the user has not chosen an image for 'img1', continue to request one
    while img1 is None:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        # If the key is 'k' then save the image and see if the user is satisfied
        if cv2.waitKey(1) & 0xFF == ord('k'):
            img1 = frame
            response = raw_input("Are you satisfied with the image show: [y/n]: ")
            if response == 'y' or response == 'Y':
                cv2.destroyAllWindows()
                break
            else:
                img1 = None
                cv2.destroyAllWindows()
                
    cols, rows, _ = img1.shape
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    
    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        cv2.imshow('original', img1)
        cv2.imshow('current', frame)
        rigid_transform, img3 = None, None
#         if len(args) > 2:
#             rigid_transform, img3 = func(img1, frame, *args[2:])
#         else:
#         rigid_transform, _ = ss_bf(img1, frame)
#         reconstruction = cv2.warpAffine(frame, cv2.invertAffineTransform(rigid_transform), (cols, rows))
#         cv2.imshow('reconstruction', reconstruction)
            
        rigid_transform, img3 = func(img1, frame, 'SIFT', True)
        cv2.imshow('SIFT', img3)
        try:
            cv2.imshow('attempted reconstruction', img3)
            reconstruction = cv2.warpAffine(frame_gray, cv2.invertAffineTransform(rigid_transform), (cols, rows))
            cv2.imshow('attempted reconstruction', reconstruction)
        except Exception as e:
            print e
            print rigid_transform
            blank = np.zeros((rows, cols, 3), np.uint8)
            blank = cv2.putText(blank, 'No img found', (10, rows/2-25), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('attempted reconstruction', blank)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

        
    
    
if __name__ == '__main__':
    # Load the images into RGB format
    img1 = cv2.imread('/home/chris/Pictures/eiffel2.jpg', 1)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    pts1 = np.float32([ [50, 50], [200, 100], [75, 200] ])
    pts2 = np.float32([ [35, 100],[200, 150], [150, 250] ])
    rows, cols, _ = img1_rgb.shape
    
    M = cv2.getAffineTransform(pts1, pts2)
#     print M
    img2_rgb = cv2.warpAffine(img1_rgb, M, (cols, rows))
    
#     build_pretty_thing(img1_rgb, img2_rgb)
#     chessboard()

#     rt, img3 = ss_bf(img1_rgb, img2_rgb, 'SIFT', True)
#     cv2.imshow('img', img3)
    cv2.waitKey(0)
#     orb_bf(img1_rgb, img2_rgb, False)
#     orb_flann(img1_rgb, img2_rgb, False)

    reconstruct_realtime(ss_bf)
    
