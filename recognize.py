#------------------------------------------------------------
# SEGMENT, RECOGNIZE and COUNT fingers from a video sequence
#------------------------------------------------------------

# organize imports
from collections import deque
import time
from imutils.video import VideoStream
import argparse
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())
greenLower = (110, 50, 50)
greenUpper = (130, 255, 255)
 
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

# global variables
bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=20):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # count the number of fingers
                fingers = count(thresholded, segmented)

                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)

##                # draw the segmented hand
##                vs = VideoStream(src=0).start()
##                # grab the current frame
##                frame = vs.read()
##         
##                # handle the frame from VideoCapture or VideoStream
##                frame = frame[1] if args.get("video", False) else frame
##         
##                # if we are viewing a video and we did not grab a frame,
##                # then we have reached the end of the video
##                if frame is None:
##                        break
##         
##                # resize the frame, blur it, and convert it to the HSV
##                # color space
##                frame = imutils.resize(frame, width=600)
##                blurred = cv2.GaussianBlur(frame, (11, 11), 0)
##                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
##         
##                # construct a mask for the color "green", then perform
##                # a series of dilations and erosions to remove any small
##                # blobs left in the mask
##                mask = cv2.inRange(hsv, greenLower, greenUpper)
##                mask = cv2.erode(mask, None, iterations=2)
##                mask = cv2.dilate(mask, None, iterations=2)
##         
##                # find contours in the mask and initialize the current
##                # (x, y) center of the ball
##                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
##                        cv2.CHAIN_APPROX_SIMPLE)
##                cnts = imutils.grab_contours(cnts)
##                center = None
##                        # only proceed if at least one contour was found
##                if len(cnts) > 0:
##                        # find the largest contour in the mask, then use
##                        # it to compute the minimum enclosing circle and
##                        # centroid
##                        c = max(cnts, key=cv2.contourArea)
##                        ((x, y), radius) = cv2.minEnclosingCircle(c)
##                        M = cv2.moments(c)
##                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
##         
##                        # only proceed if the radius meets a minimum size
##                        if radius > 10:
##                                # draw the circle and centroid on the frame,
##                                # then update the list of tracked points
##                                cv2.circle(frame, (int(x), int(y)), int(radius),
##                                        (0, 255, 255), 2)
##                                cv2.circle(frame, center, 5, (0, 0, 255), -1)
##                                pts.appendleft(center)
##                        # loop over the set of tracked points
##                for i in np.arange(1, len(pts)):
##                        # if either of the tracked points are None, ignore
##                        # them
##                        if pts[i - 1] is None or pts[i] is None:
##                                continue
##         
##                        # check to see if enough points have been accumulated in
##                        # the buffer
##                        if counter >= 10 and i == 1 and pts[-10] is not None:
##                                # compute the difference between the x and y
##                                # coordinates and re-initialize the direction
##                                # text variables
##                                dX = pts[-10][0] - pts[i][0]
##                                dY = pts[-10][1] - pts[i][1]
##                                (dirX, dirY) = ("", "")
##         
##                                # ensure there is significant movement in the
##                                # x-direction
##                                if np.abs(dX) > 20:
##                                        dirX = "East" if np.sign(dX) == 1 else "West"
##         
##                                # ensure there is significant movement in the
##                                # y-direction
##                                if np.abs(dY) > 20:
##                                        dirY = "North" if np.sign(dY) == 1 else "South"
##         
##                                # handle when both directions are non-empty
##                                if dirX != "" and dirY != "":
##                                        direction = "{}-{}".format(dirY, dirX)
##         
##                                # otherwise, only one direction is non-empty
##                                else:
##                                        direction = dirX if dirX != "" else dirY
##                                # otherwise, compute the thickness of the line and
##                        # draw the connecting lines
##                        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
##                        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
##         
##                # show the movement deltas and the direction of movement on
##                # the frame
##                cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
##                        0.65, (0, 0, 255), 3)
##                cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
##                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
##                        0.35, (0, 0, 255), 1)
##         
##                # show the frame to our screen and increment the frame counter
##                cv2.imshow("Frame", frame)
##                key = cv2.waitKey(1) & 0xFF
##                counter += 1
##         
##                # if the 'q' key is pressed, stop the loop
##                if key == ord("q"):
##                        break

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        
        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
