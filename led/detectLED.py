# LED detector 
# file detectLED.py
# Written by Nopparuj Saipong
# Last Updated: May 23, 2016

import cv2
import numpy as np
import glob

# We will be trying to isolate colors in the images out of the images at this
# point.We need to convert the images to HSV color space then try to isolate
# each of the LED color from the background.
# 1) We will then apply a thresholding fuunction to it to eliminate what is
# not wanted

H_MIN = 0
H_MAX = 256
S_MIN = 0
S_MAX = 256
V_MIN = 0
V_MAX = 256

MIN_OBJECT_AREA = 10
MAX_OBJECT_AREA = 100

windowName = "Original Image"
windowName1 = "HSV Image"
windowName2 = "Thresholded Image"
windowName3 = "After Morphological Operations"
windowName4 = "After Guassian Blur Operations"
trackbarWindowName = "Trackbars"

def nothing(x):
    pass

def createBars():
    ''' Create the sliders to filter out certain color in HSV image. 
    '''
    cv2.namedWindow(trackbarWindowName,0)
    cv2.createTrackbar("H_MIN",trackbarWindowName,H_MIN,H_MAX,nothing )
    cv2.createTrackbar("H_MAX",trackbarWindowName,H_MAX,H_MAX,nothing )
    cv2.createTrackbar("S_MIN",trackbarWindowName,S_MIN,S_MAX,nothing )
    cv2.createTrackbar("S_MAX",trackbarWindowName,S_MAX,S_MAX,nothing )
    cv2.createTrackbar("V_MIN",trackbarWindowName,V_MIN,V_MAX,nothing )
    cv2.createTrackbar("V_MAX",trackbarWindowName,V_MAX,V_MAX,nothing )


def morphOps(thresh):
    '''
    Get rid of white noise using dilation and erosion.
    Current kernal right now is used by 5px by 5px rectangle. 
    '''
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    '''
    #create structuring element that will be used to "dilate" and "erode" image.
    #the element chosen here is a 3px by 3px rectangle
    erodeElement = getStructuringElement( MORPH_RECT,Size(3,3))
    #dilate with larger element so make sure object is nicely visible
    dilateElement = getStructuringElement( MORPH_RECT,Size(8,8))
    cv2.erode(thresh,thresh,erodeElement)
    cv2.erode(thresh,thresh,erodeElement)
    cv2.dilate(thresh,thresh,dilateElement)
    cv2.dilate(thresh,thresh,dilateElement)
    '''    
    
def guassianBlur(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    
def drawObject(centers, frame):
    i = 0
    while (i < centers.size()):
        cv2.circle(frame,centers[i],20,(0,255,0),2)
        i = i + 1 
           
def main():
    
    trackObjects = False
    useMorphOps = True
    
    #//x and y values for the location of the object
    #int x=0, y=0;
    
    # create slider bars for HSV filtering
    createBars()
    
    # Video capture object to acquire webcam feed
    capture = cv2.VideoCapture(1)
    
    #//set height and width of capture frame
    #capture.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
    #capture.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);
    
    if(not capture.isOpened()):
        capture.open()
    
    # start an infinite loop where webcam feed is copied to cameraFeed matrix
    # all of our operations will be performed within this loop
    while(capture.isOpened()):
        
        # capture the frame
        ret, cameraFeed = capture.read()
        
        # Get trackbar positions
        H_MIN = cv2.getTrackbarPos("H_MIN",trackbarWindowName)
        H_MAX = cv2.getTrackbarPos("H_MAX",trackbarWindowName)
        S_MIN = cv2.getTrackbarPos("S_MIN",trackbarWindowName)
        S_MAX = cv2.getTrackbarPos("S_MAX",trackbarWindowName)
        V_MIN = cv2.getTrackbarPos("V_MIN",trackbarWindowName)
        V_MAX = cv2.getTrackbarPos("V_MAX",trackbarWindowName)
        
        #cv2.imshow(windowName,cameraFeed)
        
        # convert frame from BGR to HSV colorspace
        if (ret == True):
            HSV = cv2.cvtColor(cameraFeed,cv2.COLOR_BGR2HSV)
            grey = cv2.cvtColor(cameraFeed,cv2.COLOR_BGR2GRAY)
            
            # filter HSV image between values and store filtered image to
            # threshold matrix
            lower_mask = np.array([H_MIN,S_MIN,V_MIN])
            higher_mask = np.array([H_MAX,S_MAX,V_MAX])
            mask = cv2.inRange(HSV,lower_mask,higher_mask)
            threshold_morph = cv2.bitwise_and(grey, grey, mask = mask)
            threshold_blur  = cv2.bitwise_and(grey, grey, mask = mask)
            # we want to perform bitwise masking here from HSV to greyscale so that
            # we can extract the 'center' position using weighted intensity        
            
            #perform morphological operations on thresholded image to eliminate noise
            #and emphasize the filtered object(s)
            if(useMorphOps):
                morphOps(threshold_morph)
                guassianBlur(threshold_blur)
                
            #pass in thresholded frame to our object tracking function
            #this function will return the x and y coordinates of the
            #filtered object
            #if(trackObjects)
            #	trackFilteredObject(x,y,threshold,cameraFeed);
            
            # count the number of blobs (aka the leds we could detect)
            contours,hierarchy = cv2.findContours(threshold_morph.clone(),cv2.CV_RETR_EXTERNAL 
                ,cv2.CV_CHAIN_APPROX_NONE)
            
            # identify the blobs in the image
            numPoints = 0 # the number of detected LEDs
            distorted_points = []
            refArea = 0
            i = 0
            while (i < contours.size()):
                area = cv2.contourArea(contours[i]) # get the area
                #rect = cv2.boundingRect(contours[i]) # bounding rectangle box
                #radius = (rect.width + rect.height) / 4 # average radius
                mu = cv2.moments(contours[i])
               
                if((area>MIN_OBJECT_AREA) and (area<MAX_OBJECT_AREA) and (area>refArea)):
                    cx = mu.m10 / mu.m00
                    cy = mu.m01 / mu.m00
                    #mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00) + cv::Point2f(ROI.x, ROI.y);
                    distorted_points.append([cx,cy]) # store location as tuple
                    numPoints = numPoints + 1
    
                i = i + 1
                
            centers = np.array(distorted_points)     
       
            # show frames 
            cv2.imshow(windowName4,threshold_blur);
            cv2.imshow(windowName3,threshold_morph);
            cv2.imshow(windowName2,mask);
            cv2.imshow(windowName,cameraFeed);
            cv2.imshow(windowName1,HSV);
    
        # allow 30 ms for screen to refresh and wait for ESC
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break	
    
    capture.release()
    cv2.destroyAllWindows()

    '''
    # Threshold the image (currently at 
    img = cv2.imread('green.png',1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img,254,255,cv2.THRESH_TOZERO)
        
    # Gaussian blur the image to remove any noise
    # current Guassian Kernal is 5,5. If it is 0 then gaussian_sigma will be used
    blur = cv2.GaussianBlur(thresh1,(5,5),0)
    
    cv2.imshow('Blurred and threshed',blur)
    cv2.imshow('Original',img)
    '''
    
    if __name__ == '__main__':
        main()