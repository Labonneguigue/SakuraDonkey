import argparse
import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from donkeycar.parts.calibration import Calibration


class CalibratedAlgorithm(object):
    '''
    Any algorithm that needs camera calibration data will be
    able to fetch the calibration data using this class constructor
    '''
    def __init__(self, camera):
        super(CalibratedAlgorithm, self).__init__()
        calibration = Calibration()
        self.mtx, self.dist = calibration.load(camera)

    def getCalibration():
        '''
        Returns the camera calibration
        '''
        return self.mtx, self.dist

class LanesDetector(CalibratedAlgorithm):
    '''
    This node takes as input an image and detects the left and right lines that
    constitute the lane the car is in.

    Args:
        camera: Name of the camera currently used. The string will be used to
            fetch calibration data.
        img_size: The lanedetection algorithm needs to know the size of the image.
    '''
    def __init__(self, camera, config, img_size=(720, 1280)):
        super(LanesDetector, self).__init__(camera)
        # Image height and width
        self.imageHeight = img_size[0]
        self.imageWidth = img_size[1]
        self.cfg = config
        print(self.cfg)
        self.reset()

    def reset(self):
        self.imageCounter = 0
        # Indicates whether the sliding window histogram needs to
        # be computed
        self.lanesDetected = None
        # Perspective matrices
        self.perspectiveMat = None
        self.inversePerspectiveMat = None
        # Set the number of sliding windows
        self.nbSlidingWindows = 9
        # Set the height of the sliding windows
        self.slidingWindowHeight = np.int(self.imageHeight/self.nbSlidingWindows)
        # Base of 2 sliding windows
        self.leftx_base = None
        self.rightx_base = None
        # Linear plot of the Y axis
        self.ploty = np.linspace(0, self.imageHeight - 1, self.imageHeight)
        # Polynomials for both lanes
        self.leftPolynomialFit = None
        self.rightPolynomialFit = None
        # X component of these Polynomials
        self.left_fitx = None
        self.right_fitx = None
        # Positions of the non zero pixels in the image
        self.nonzerox = None
        self.nonzeroy = None
        # Empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []
        # Set the width of the windows +/- margin
        self.margin = P.parameters['margin']
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        # Image to write on and visualize the results
        self.outputImage = None
        ######
        # Radius of Curvature calculation
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Left and Right curvature radius
        self.left_curverad = 0
        self.right_curverad = 0
        # Position of the car away from the center of the lane (m)
        self.carPosition = 0
        self.carSide = "Uninitialized" # "Left " or "Right"

    def InitPerspectiveMatrix(self, img):
        '''
        Order of the points
                1 ---- 2        <- parameters['orig_points_y']
               /        \
              /          \
             /            \
            4 ------------ 3

        Args:
            img: An image. Could be the first one of the recording. If the size
            of the image changes afterwards, this function needs to be called
            again
        '''
        src = np.float32(
            [[P.parameters['orig_points_x'][0], P.parameters['orig_points_y']],
             [P.parameters['orig_points_x'][1],  P.parameters['orig_points_y']],
             [P.parameters['orig_points_x'][2],  img.shape[0]],
             [P.parameters['orig_points_x'][3],  img.shape[0]]])
        dst = np.float32(
            [[(img.shape[1] / 4),    P.parameters['detection_distance']],
             [ (img.shape[1] * 3 / 4),P.parameters['detection_distance']],
             [ (img.shape[1] * 3 / 4), img.shape[0]],
             [ (img.shape[1] / 4),     img.shape[0]]])
        self.perspectiveMat = cv2.getPerspectiveTransform(src, dst)
        self.inversePerspectiveMat = cv2.getPerspectiveTransform(dst, src)

    def MergeChannels(self, imgA, imgB, imgC=None):
        '''
        Combine multiple images

        Args:
            imgA: A single dimension image to be merge
            imgB: A single dimension image to be merge
            imgC: An optional single dimension image to be merge

        Returns:
            Returns the bitwise OR operation on the 2 or 3 provided
            images
        '''
        if imgC is None:
            return cv2.bitwise_or(imgA, imgB)
        else:
            return cv2.bitwise_or(cv2.bitwise_or(imgA, imgB), imgC)

    def LoadImage(self, fname = 'test_images/test5.jpg'):
        '''
        Load and returns an image in the RGB colour space

        Args:
            fname: Path to an image file
        
        Returns:
            The image as an array in the RGB colour space 
        '''
        return mpimg.imread(fname)

    def ColorChannelBinarization(self, s_channel, s_thresh=(170, 255)):
        '''
        Binarized the S channel within the specified threshold.
        
        Args:
            s_channel: Colour channel (unidimensional)
            s_thresh: Each pixel with the 2 specified values will be turned on

        Returns:
            A mono channel image of the passing pixels
        '''
        s_binary = np.zeros_like(s_channel)
        _, s_binary = cv2.threshold(s_channel.astype('uint8'), s_thresh[0], s_thresh[1], cv2.THRESH_BINARY)
        return s_binary

    def SobelBinarization(self, l_channel, s_thresh=(20, 100), direction='x'):
        '''
        Performs Sobel derivative function in either x or y direction
        Expects an image L channel from its HLS representation.
        '''
        if direction == 'x':
            sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        else:
            sobel = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
        abs_sobel = np.absolute(sobel) # Absolute derivative to accentuate lines away from horizontal/vertical
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # Threshold x gradient
        sbinary = np.zeros_like(scaled_sobel)
        #sbinary[(scaled_sobel >= s_thresh[0]) & (scaled_sobel <= s_thresh[1])] = 1
        _, sbinary = cv2.threshold(scaled_sobel.astype('uint8'), s_thresh[0], s_thresh[1], cv2.THRESH_BINARY)
        return sbinary


    def StackImages(self, imgA, imgB, imgC=None):
        '''
        Stack 3 2D images into a 3D image.
        The 2 images need to have the same shape.
        '''
        assert(imgA.shape == imgB.shape)
        if imgC==None:
            return np.dstack((np.zeros_like(imgA), imgA, imgB))
        else:
            return np.dstack((imgA, imgB, imgC))

    def DisplayAndSave2Images(self, imgA, imgB, name, title='Pipeline Result', grayscale=False):
        '''
        Display 2 images side by side and save the figure
        for a quick comparison
        '''
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(imgA)
        ax1.set_title('Original Image', fontsize=40)
        if grayscale:
            ax2.imshow(imgB, cmap=plt.cm.gray)
        else:
            ax2.imshow(imgB)
        ax2.set_title(title, fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig('output_images/' + name)


    def UpdateCurvatureRadius(self):
        '''
        Calculate the curvature radius of the 2 lines of the current lane
        and assigns them to the class member keeping track of them.
        
            NOTE: This function exist and is different from CalculateCurvatureRadius()
            to give the possibility to calculate lane curvature radius without updating
            the class members
        '''
        self.left_curverad, self.right_curverad = self.CalculateCurvatureRadius()

    def CalculateCurvatureRadius(self, leftFitX=None, rightFitX=None):
        '''
        Function that returns the radius of curvature of 2 polynomials
        Generic to be able to test new polynomial fits for accuracy
        before setting them as class attributes and plot them.
        '''
        if leftFitX is None:
            leftFitX = self.left_fitx
        if rightFitX is None:
            rightFitX = self.right_fitx
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = self.imageHeight - 1
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty*self.ym_per_pix, leftFitX*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*self.ym_per_pix, rightFitX*self.xm_per_pix, 2)
        # Calculate the radius of curvature of both polynomials
        leftCurverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        rightCurverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        return leftCurverad, rightCurverad

    def UpdateLateralLanePosition(self):
        '''
        Returns the position of the car away from the center of the
        road in meters. A positive value is on the right of center
        I am using the value at the bottom of the image of each fitted
        polynomial to evaluate the position of car compared to the
        lines. I am assuming the camera is mounted in the center of the
        windshield.
        '''
        roadCenter = (self.left_fitx[-1] + self.right_fitx[-1]) / 2
        # Conversion from pixels to meters using Udacity conversion data
        self.carPosition = (roadCenter - self.imageWidth/2)*self.xm_per_pix
        if (self.carPosition < 0):
            self.carSide = "Right"
        else:
            self.carSide = "Left "

    def BlindSlidingWindowsHistogram(self, binary_warped):
        '''
        This function runs without prior knowledge of where the lane are.
        It first creates a horizontal histogram of the lower half of the
        binarized pixel. 
        Then, on each half (horizontally) the spikes give a hint where
        the lanes start at the bottom of the image.
        Lastly, rectangles are stacked onto another iteratively to follow
        the binarized pixels distribution supposedly representing the lane.

        Args:
            binary_warped: Mono channel binarized image.

        Returns:
            The input image with the boxes and lines overlaid
        '''
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2.):,:], axis=0)
        # Create an output image to draw on and visualize the result
        outputImage = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        if self.leftx_base is None:
            self.leftx_base = np.argmax(histogram[:midpoint])
            self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        else:
            self.leftx_base = int(((self.leftx_base* 9) + np.argmax(histogram[:midpoint])) / 10)
            self.rightx_base = int(((self.rightx_base * 9) + (np.argmax(histogram[midpoint:]) + midpoint)) / 10)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        # Empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nbSlidingWindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*self.slidingWindowHeight
            win_y_high = binary_warped.shape[0] - window*self.slidingWindowHeight
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(outputImage,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(outputImage,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            # print("good_left_inds")
            # print(good_left_inds)
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)

            cv2.line(outputImage, (leftx_current, win_y_low), (leftx_current, win_y_high), (0, 255, 255), thickness=2)
            cv2.line(outputImage, (rightx_current, win_y_low), (rightx_current, win_y_high), (0, 255, 255), thickness=2)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            else:
                pass
                # good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy <= win_y_high) & (self.nonzerox >= leftx_current-2) & (self.nonzerox <= leftx_current+2))[0]
                # print(good_left_inds.shape)
                # # for y in range(win_y_low, win_y_high):
                # #     self.left_lane_inds.append((leftx_current-2, y))
                # #     self.left_lane_inds.append((leftx_current-1, y))
                # #     self.left_lane_inds.append((leftx_current, y))
                # #     self.left_lane_inds.append((leftx_current+1, y))
                # #     self.left_lane_inds.append((leftx_current+2, y))
                # self.left_lane_inds.append(good_left_inds)
                # print(good_left_inds)

            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))
            else:
                pass
                # for y in range(win_y_low, win_y_high):
                #     self.right_lane_inds.append((rightx_current, y))
                #     #print((rightx_current, y))
                # good_right_inds = (self.nonzeroy >= win_y_low) & (self.nonzeroy <= win_y_high) & (self.nonzerox >= rightx_current-2) & (self.nonzerox <= rightx_current+2)
                # self.right_lane_inds.append(good_right_inds)


        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        print(leftx)
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        self.leftPolynomialFit = None
        self.rightPolynomialFit = None
        # Fit a second order polynomial to each
        self.leftPolynomialFit = np.polyfit(lefty, leftx, 2)
        self.rightPolynomialFit = np.polyfit(righty, rightx, 2)
        #return self.outputImage
        self.lanesDetected = True
        return outputImage

    def DetectionFromPreviousPolynomial(self, binary_warped):
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        self.left_lane_inds = ((self.nonzerox > (self.leftPolynomialFit[0]*(self.nonzeroy**2) + \
                                       self.leftPolynomialFit[1]*self.nonzeroy + \
                                       self.leftPolynomialFit[2] - self.margin))
                             & (self.nonzerox < (self.leftPolynomialFit[0]*(self.nonzeroy**2) + \
                                       self.leftPolynomialFit[1]*self.nonzeroy + \
                                       self.leftPolynomialFit[2] + self.margin)))
        self.right_lane_inds = ((self.nonzerox > (self.rightPolynomialFit[0]*(self.nonzeroy**2) + \
                                        self.rightPolynomialFit[1]*self.nonzeroy + \
                                        self.rightPolynomialFit[2] - self.margin)) \
                              & (self.nonzerox < (self.rightPolynomialFit[0]*(self.nonzeroy**2) + \
                                        self.rightPolynomialFit[1]*self.nonzeroy + \
                                        self.rightPolynomialFit[2] + self.margin)))

        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]
        # Fit a second order polynomial to each
        self.leftPolynomialFit = np.polyfit(lefty, leftx, 2)
        self.rightPolynomialFit = np.polyfit(righty, rightx, 2)

    def PolynomialFitAnalysis(self):
        '''
        Evaluate the accuracy likelihood of the polynomial fit
        for both lanes. Discards it if too unlikely.
        Idea: if the radius of curvature becomes significantly
        smaller (of a factor 8-10) from one frame to the other
        and also bellow a certain threshold, I take the decision
        to discard the polyfit and keep the previous one.
        '''
        # Generate x and y values for plotting
        currentLeftFitx = self.leftPolynomialFit[0]*self.ploty**2 + self.leftPolynomialFit[1]*self.ploty + self.leftPolynomialFit[2]
        currentRightFitx = self.rightPolynomialFit[0]*self.ploty**2 + self.rightPolynomialFit[1]*self.ploty + self.rightPolynomialFit[2]
        currentLeftRad, currentRightRad = self.CalculateCurvatureRadius(currentLeftFitx, currentRightFitx)
        if self.left_fitx is None:
            self.left_fitx = self.leftPolynomialFit[0]*self.ploty**2 + self.leftPolynomialFit[1]*self.ploty + self.leftPolynomialFit[2]
        if self.right_fitx is None:
            self.right_fitx = self.rightPolynomialFit[0]*self.ploty**2 + self.rightPolynomialFit[1]*self.ploty + self.rightPolynomialFit[2]

        if 0:
            '''TODO: Remove later'''
            self.left_fitx = self.leftPolynomialFit[0]*self.ploty**2 + self.leftPolynomialFit[1]*self.ploty + self.leftPolynomialFit[2]
            self.right_fitx = self.rightPolynomialFit[0]*self.ploty**2 + self.rightPolynomialFit[1]*self.ploty + self.rightPolynomialFit[2]

        if 1:
            if currentRightRad / currentLeftRad > 7:
                self.right_curverad = currentRightRad
                self.right_fitx = currentLeftFitx
            if currentLeftRad / currentRightRad > 7:
                self.left_curverad = currentLeftRad
                self.left_fitx = currentLeftFitx

        '''TODO: Enable later'''
        if 0:
            if (self.left_curverad / currentLeftRad >= 8) & (currentLeftRad <= 650):
                #New polynomial discarded
                self.lanesDetected = False
            else:
                self.left_curverad = currentLeftRad
                self.left_fitx = currentLeftFitx
            if (self.right_curverad / currentRightRad >= 8) & (currentRightRad <= 650):
                #New polynomial discarded
                self.lanesDetected = False
            else:
                self.right_curverad = currentRightRad
                self.right_fitx = currentLeftFitx

    def VisualizeHistogramPolynomial(self, outputImage):
        outputImage[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        outputImage[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
        if (self.left_fitx.astype('uint16') < self.imageWidth).all():
            outputImage[self.ploty.astype('uint16'), self.left_fitx.astype('uint16')] = [255, 255, 0]
        else:
            self.lanesDetected = False
        if (self.right_fitx.astype('uint16') < self.imageWidth).all():
            outputImage[self.ploty.astype('uint16'), self.right_fitx.astype('uint16')] = [255, 255, 0]
        else:
            self.lanesDetected = False

        return outputImage, self.ploty, self.left_fitx, self.right_fitx

    def VisualizePolynomial(self, binary_warped):
        outputImage = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(outputImage)
        # Color in left and right line pixels
        outputImage[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        outputImage[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        self.left_fitx = self.leftPolynomialFit[0]*self.ploty**2 + self.leftPolynomialFit[1]*self.ploty + self.leftPolynomialFit[2]
        self.right_fitx = self.rightPolynomialFit[0]*self.ploty**2 + self.rightPolynomialFit[1]*self.ploty + self.rightPolynomialFit[2]

        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-self.margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+self.margin, self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-self.margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+self.margin, self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        if (self.left_fitx.astype('uint16') < self.imageWidth).all():
            outputImage[self.ploty.astype('uint16'), self.left_fitx.astype('uint16')] = [255, 255, 0]
        if (self.right_fitx.astype('uint16') < self.imageWidth).all():
            outputImage[self.ploty.astype('uint16'), self.right_fitx.astype('uint16')] = [255, 255, 0]

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(outputImage, 1, window_img, 0.25, 0)

        # plt.close('all')
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        #
        # plt.imshow(result)
        # ax.plot(self.left_fitx, self.ploty, color='yellow')
        # ax.plot(self.right_fitx, self.ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        #fig.canvas.draw()

        # Now we can save it to a numpy array.
        #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #print(data.shape)
        return result, self.ploty, self.left_fitx, self.right_fitx

    def PerspectiveTransform(self, img, perspectiveMatrix):
        '''
        Change the perspective of the image
        '''
        return cv2.warpPerspective(img, perspectiveMatrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def Binarization(self, img):
        '''
        This processing pipeline is composed of the following steps:
         - Image undistortion
         - Image Binarization
         - Perspective Transform
        This needs to be applied to every frame before starting detecting the lane lines
        '''
        # Convert to HLS color space and separate the L and S channels
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        SobelBinary = self.SobelBinarization(l_channel)  # sobel along x
        WhiteBinary = self.ColorChannelBinarization(l_channel, (200, 255))

        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
        # lower = np.uint8([ 10,   0, 100])
        # upper = np.uint8([ 40, 255, 255])
        # YellowBinary = cv2.inRange(hsv, lower, upper)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
        yellow_dark = np.array([10, 127, 127], dtype=np.uint8)
        yellow_light = np.array([35, 255, 255], dtype=np.uint8)
        YellowBinary = cv2.inRange(hsv, yellow_dark, yellow_light)
        assert(WhiteBinary.shape == YellowBinary.shape)
        # return MergeChannels(SobelBinary, WhiteBinary, YellowBinary)
        return self.MergeChannels(SobelBinary, WhiteBinary, YellowBinary)

    def OverlayDetectedLane(self, image, unwarp=True):
        '''
        Create an image to draw the detected lane on, unwarp it
        and use cv2.addWeighted() function to overlay it on top
        of the actual image.
        '''
        #Create empty image to draw the lane on
        lane = np.zeros_like(image).astype(np.uint8)

        # Cast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the empty image
        cv2.fillPoly(lane, np.int_([pts]), (255, 0, 0))

        if unwarp:
            # Warp the empty back to original image space using inverse perspective matrix
            output = cv2.warpPerspective(lane, self.inversePerspectiveMat, (self.imageWidth, self.imageHeight))
        else:
            #If no unwarp, the detected lane is still in topDownView mode
            output = lane
        # Combine the result with the original image
        return cv2.addWeighted(image, 1, output, 0.3, 0)

    def AggregateViews(self, views):
        '''
        Creates an aggregated image for a better display
        '''
        H = 1080 #self.imageHeight      #Height of the output image
        W = 1920 #self.imageWidth       #Width of the output image
        aggregateViews = np.zeros((H, W, 3), dtype=np.uint8)
        h2 = int(H/2)
        w2 = int(W/2)
        # Input image with overlaid detected lane -- 1
        aggregateViews[0:h2, 0:w2] = cv2.resize(views[0],(w2,h2), interpolation=cv2.INTER_AREA)
        # Binarized image -- 2
        view2 = np.dstack((views[1], views[1], views[1]))
        aggregateViews[0:h2, w2:W] = cv2.resize(view2,(w2,h2), interpolation=cv2.INTER_AREA)
        # topDownView overlaid with detected lane -- 3
        aggregateViews[h2:H, 0:w2] = cv2.resize(views[2],(w2,h2), interpolation=cv2.INTER_AREA)
        # Warped perspective and polynomial fit of the lines -- 4
        aggregateViews[h2:H, w2:W] = cv2.resize(views[3],(w2,h2), interpolation=cv2.INTER_AREA)

        return aggregateViews

    def RenderText(self, image):
        '''
        Renders information on the output image
        '''
        font = cv2.FONT_HERSHEY_DUPLEX
        fontColor = (255, 255, 255)
        curvatureL = "Radius of Curvature - Left line : " + str(self.left_curverad) + " m"
        curvatureR = "Radius of Curvature - Right line : " + str(self.right_curverad) + " m"
        cv2.putText(image, curvatureL, (30, 40), font, 1, fontColor, 2, cv2.LINE_AA)
        cv2.putText(image, curvatureR, (30, 80), font, 1, fontColor, 2, cv2.LINE_AA)
        lateralDisplacement = "Distance away from lane center (" + self.carSide + ") : " + str(self.carPosition) + " (m)"
        cv2.putText(image, lateralDisplacement, (30, 120), font, 1, fontColor, 2, cv2.LINE_AA)
        frameNumber = "Frame number : " + str(self.imageCounter)
        cv2.putText(image, frameNumber, (30, 160), font, 1, fontColor, 2, cv2.LINE_AA)


    def ProcessImage(self, image, key_frame_interval=20, cache_length=10):
        assert(image.shape[0] == self.imageHeight)
        assert(image.shape[1] == self.imageWidth)
        #TODO: Maybe change that
        # Initialisation on first frame
        if self.perspectiveMat is None:
            InitPerspectiveMatrix(image)
        
        self.imageCounter += 1

        undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        topDownView = self.PerspectiveTransform(undistorted, self.perspectiveMat)
        topDownViewBinarized = self.Binarization(topDownView)

        binary = self.Binarization(undistorted)
        #topDownViewBinarized = self.PerspectiveTransform(binary, PersMat)
        #topDownView = self.PerspectiveTransform(undistorted, PersMat)

        # result = self.BlindSlidingWindowsHistogram(topDownViewBinarized)
        # self.PolynomialFitAnalysis()
        # result, ploty, l_fit, r_fit = self.VisualizeHistogramPolynomial(result)

        if self.lanesDetected:
            self.DetectionFromPreviousPolynomial(topDownViewBinarized)
            self.PolynomialFitAnalysis()
            result, ploty, l_fit, r_fit = self.VisualizePolynomial(topDownViewBinarized)
        else:
            # Only when the polynomial is not defined or lost
            result = self.BlindSlidingWindowsHistogram(topDownViewBinarized)
            self.PolynomialFitAnalysis()
            result, ploty, l_fit, r_fit = self.VisualizeHistogramPolynomial(result)

        self.UpdateLateralLanePosition()
        self.UpdateCurvatureRadius()
        self.RenderText(image)
        overlaidLane = self.OverlayDetectedLane(image)
        topDownViewOverlaidLane = self.OverlayDetectedLane(topDownView, unwarp=False)

        # Create an output image composed of different views for debug and illustration purposes
        if P.parameters['debug_output']:
            result = self.AggregateViews([overlaidLane, binary, result, topDownViewOverlaidLane])
        else:
            result = overlaidLane

        return result

    def OutputImages(self, image, key_frame_interval=20, cache_length=10):
        binary = self.ProcessingPipeline(image)
        mpimg.imsave("videos_images/videoImage-" + str(self.imageCounter) + ".jpg", image, cmap=plt.cm.gray)
        mpimg.imsave("videos_images/videoImage-" + str(self.imageCounter) + "-binarized.jpg", binary, cmap=plt.cm.gray)
        self.imageCounter += 1
        return binary

    
    def run(self, image=None, reset=False):
        if reset is True:
            print("LaneDetector reset ...")
            self.reset()
        return self.ProcessImage(image)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Detects lane lines on the road. Can be used online (using the camera) or offline (using a recording.')
    parser.add_argument('-c', '--camera', action="store_true", help='uses the camera as an input the the algorithm')
    args = parser.parse_args()
    print(args.camera)

    lanesDetector = LanesDetector("carla")

    print('Processing video ... ' + P.parameters['videofile_in'])
    vfc = VideoFileClip(P.parameters['videofile_in']).subclip(20, 24)
    if P.parameters['output_video_as_images']:
        detected_vid_clip = vfc.fl_image(lanesDetector.OutputImages)
    else:
        detected_vid_clip = vfc.fl_image(lanesDetector.ProcessImage)
    detected_vid_clip.write_videofile(P.parameters['videofile_out'], audio=False)
