'''
LED detector 
file: detectLED.py
function: detect LED and return center points of the led
Written by Nopparuj Saipong
Last Updated: May 31, 2016
'''

import cv2
import numpy as np
import glob

# We will be trying to isolate colors in the images out of the images at this
# point.We need to convert the images to HSV color space then try to isolate
# each of the LED color from the background.
# 1) We will then apply a thresholding fuunction to it to eliminate what is
# not wanted

class Filter:
    V_MIN = 253
    V_MAX = 255
    S_MIN = 0
    def __init__(self,color):
        if (color == "red"):
            self.H_MIN = 0
            self.H_MAX = 50
            self.S_MAX = 12
        if (color == "green"):
            self.H_MIN = 83
            self.H_MAX = 104 
            self.S_MAX = 5
        if (color == "blue"):
            self.H_MIN = 80
            self.H_MAX = 93
            self.S_MAX = 18


H_MIN = 0
H_MAX = 255
S_MIN = 0
S_MAX = 255
V_MIN = 0
V_MAX = 255

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
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    modifiedImg = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

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
    return modifiedImg
    
def guassianBlur(img):
    blurredImg = cv2.GaussianBlur(img,(31,31),0,0)
    return blurredImg
    
def drawObject(centers, frame):
    i = 0
    while (i < centers.size()):
        cv2.circle(frame,centers[i],20,(0,255,0),2)
        i = i + 1
        
def trackFilteredObject(cameraFeed,threshold):
    ''' 
    Returns a set of'center points' of the blob areas which met the requirement
    for tracking.
    Input - cameraFeed = the image frame taken from the camera
            threshold  = the thresholded and morphed or filtered image
    '''
    
    contours,hierarchy = cv2.findContours(np.copy(threshold),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                
    # identify the blobs in the image
    numPoints = 0 # the number of detected LEDs
    distorted_points = []
    refArea = 0
    i = 0
    while (i < len(contours)):
            area = cv2.contourArea(contours[i]) # get the area
            #rect = cv2.boundingRect(contours[i]) # bounding rectangle box
            #radius = (rect.width + rect.height) / 4 # average radius
            mu = cv2.moments(contours[i])
                
            if((area>MIN_OBJECT_AREA) and (area<MAX_OBJECT_AREA) and (area>refArea)):
                cx = mu['m10'] / mu['m00']
                cy = mu['m01'] / mu['m00']
                #mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00) + cv::Point2f(ROI.x, ROI.y);
                distorted_points.append((cx,cy)) # store location as tuple
                numPoints = numPoints + 1
        
            i = i + 1
                    
    centers = np.array(distorted_points)
    return centers    

def drawCrosshairs(coordinates,img):
    height, width = img.shape[:2]
    drawn = np.copy(img)
    for tuples in coordinates:
            x, y = tuples
            x = int(x)
            y = int(y)
            print (x,y)
            cv2.circle(drawn,(x,y),20,(0,255,0),5)
            if(y-25>0):
                cv2.line(drawn,(x,y),(x,y-25),(0,255,0),5)
            else: 
                cv2.line(drawn,(x,y),(x,0),(0,255,0),5)
            if(y+25<height):
                cv2.line(drawn,(x,y),(x,y+25),(0,255,0),5)
            else:
                cv2.line(drawn,(x,y),(x,height),(0,255,0),5)
            if(x-25>0):
                cv2.line(drawn,(x,y),(x-25,y),(0,255,0),5)
            else:
                cv2.line(drawn,(x,y),(0,y),(0,255,0),5)
            if(x+25<width):
                cv2.line(drawn,(x,y),(x+25,y),(0,255,0),5)
            else:
                cv2.line(drawn,(x,y),(width,y),(0,255,0),5)
    return drawn
                
def main():
    
    trackObjects = True
    useMorphOps = True
    color = "blue"
    
    #//x and y values for the location of the object
    #int x=0, y=0;
    
    # create slider bars for HSV filtering
    createBars()
    
    # Video capture object to acquire webcam feed
    capture = cv2.VideoCapture(0)
    
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
        
        # Currently the trackbar is disabled. To reenable,
        # user MUST recreate variables listed below of default values.
        
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
            colorFilter = Filter(color)
            #lower_mask = np.array([colorFilter.H_MIN,colorFilter.S_MIN,colorFilter.V_MIN])
            #higher_mask = np.array([colorFilter.H_MAX,colorFilter.S_MAX,colorFilter.V_MAX])
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
                threshold_morph = morphOps(threshold_morph)
                threshold_blur  = guassianBlur(threshold_blur)
                
            #pass in thresholded frame to our object tracking function
            #this function will return the x and y coordinates of the
            #filtered object
            #if(trackObjects)
            #	trackFilteredObject(x,y,threshold,cameraFeed);

            if (trackObjects):
                '''
                # count the number of blobs (aka the leds we could detect)
                contours,hierarchy = cv2.findContours(np.copy(threshold_morph),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                
                # identify the blobs in the image
                numPoints = 0 # the number of detected LEDs
                distorted_points = []
                refArea = 0
                i = 0
                while (i < len(contours)):
                    area = cv2.contourArea(contours[i]) # get the area
                    #rect = cv2.boundingRect(contours[i]) # bounding rectangle box
                    #radius = (rect.width + rect.height) / 4 # average radius
                    mu = cv2.moments(contours[i])
                
                    if((area>MIN_OBJECT_AREA) and (area<MAX_OBJECT_AREA) and (area>refArea)):
                        cx = mu['m10'] / mu['m00']
                        cy = mu['m01'] / mu['m00']
                        #mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00) + cv::Point2f(ROI.x, ROI.y);
                        distorted_points.append((cx,cy)) # store location as tuple
                        numPoints = numPoints + 1
        
                    i = i + 1
                    
                centers = np.array(distorted_points)    
                '''
                centers = trackFilteredObject(cameraFeed,threshold_morph)
                cameraFeed = drawCrosshairs(centers,cameraFeed)
               
                
            # show frames 
            cv2.imshow(windowName4,threshold_blur);
            cv2.imshow(windowName3,threshold_morph);
            cv2.imshow(windowName2,mask);
            cv2.imshow(windowName,cameraFeed);
            #cv2.imshow(windowName1,HSV);
    
            print centers
            
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