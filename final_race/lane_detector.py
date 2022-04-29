# This Python Script will take given brick color ranges and return isolated image of just that brick
# as well as the brick center location and orientation
from dis import dis
import cv2
import os
from cv2 import bitwise_and
import numpy as np

# Sofware flags
DEBUG = True
USE_CAMERA_FEED = False
DISPLAY_IMAGES = True

# Color isolation parameters
color_range = 10
HUE_BLUE = 113
HUE_RED_1 = 174
HUE_RED_2 = 5
HUE_GREEN = 90
HUE_YELLOW = 30
area_limit = 50000 # if sum is less than 50,000 pixels we can assume we do not see a full brick. TODO find correct area estimate
color_state = "" # Start off with empty color state
"""
NOTE Red is a special case since color goes past total range we have to split into 2 cases
Threshold one is from 169 - 179, Threshold 2 is from 0 - 10. In this case our range from middle is only 5 not 10
Then we have to combine results of both thresholds using bitwise_or() operator
"""

def display_image(label, scale, img):
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale)) 
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    cv2.imshow(label, resized_img)

#This function reads every image from the folder
#TODO: Delete when switching to live camera
def load_img_from_folder(folder_path):
    pictures = {}
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            pictures[filename] = img
    return pictures

# This function removes isolated pixels
def denoise_img(image):
    kernel = np.ones((3,3), np.uint8)
    # Erode image to remove noise, then dilate image to return it to original size  
    eroded_image_1 = cv2.erode(image, kernel, iterations=3)
    dilated_image_1 = cv2.dilate(eroded_image_1, kernel, iterations=3)
    
    # Dilate image to get rid of empty spots in brick blob, then erode image to return it to original size
    dilated_image_2 = cv2.dilate(dilated_image_1, kernel, iterations=2)
    eroded_image_2 = cv2.erode(dilated_image_2, kernel, iterations=2)
    return eroded_image_2

def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
                            [(0, height), (800, height), (380, 290)]
                        ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv2.bitwise_and(frame, mask)
    return segment

def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    print(lines)
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    if np.isnan(parameters).any():
        return np.zeros(4).astype(int)

    print("param", parameters)
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 15)
    return lines_visualize
        
def init():
    if USE_CAMERA_FEED:
        #Live Camera Information
        camera = cv2.VideoCapture(4) #4 is the port for my external camera
        # Show error if camera doesnt show up
        if not camera.isOpened():
            raise Exception("Could not open video device")
        # Set picture Frame. High quality is 1280 by 720
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        file_path = "./../media"
        images = load_img_from_folder(file_path)
        raw_img = images["track_straight.png"]
        hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
        
    while True:
        if USE_CAMERA_FEED:
            ret, raw_img = camera.read() #get current image feed from camera
            if not ret:
                print("Error. Unable to capture Frame")
                break
            hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV) # converts photo from RGB to HSV
        lower_white_hsv_range = np.array([81, 8, 212], dtype=np.uint8)
        upper_white_hsv_range = np.array([108, 55, 255], dtype=np.uint8)
        threshold = cv2.inRange(hsv, lower_white_hsv_range, upper_white_hsv_range) 

        clean_mask = denoise_img(threshold)

        lower_threshold = 50
        upper_threshold = 150

        # gray = cv2.cvtColor(threshold, cv2.COLOR_RGB2GRAY)
        # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
        blur = cv2.GaussianBlur(threshold, (5, 5), 0)
        # Applies Canny edge detector with minVal of 50 and maxVal of 150
        # canny = cv2.Canny(blur, 50, 150)

        canny_edges = cv2.Canny(blur, lower_threshold, upper_threshold, apertureSize = 3)   # Canny edge detector to make it easier for hough transform to "agree" on lines
        # cv2.imshow("Canny_Image", canny_edges)
        # display_image("Canny Edges", 0.16, canny_edges)
        cv2.waitKey(3) 

        # min_intersections = 200                           # TO DO: Play with this parameter to change sensitivity.
        # lines = cv2.HoughLines(canny_edges, 1,np.pi/180,min_intersections)     # Run Hough Transform
        

        segment = do_segment(canny_edges)
        hough = cv2.HoughLinesP(canny_edges, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 100)
        
        line_thickness = 15
        if hough is not None:
            num_lines = 0
            shape = hough.shape
            for i in range(shape[0]):                         # Plot hough over original feed
                for x1,y1,x2,y2 in hough[i]:
                    # a = np.cos(theta)
                    # b = np.sin(theta)
                    # x0 = a*rho
                    # y0 = b*rho
                    # x1 = int(x0 + 1000*(-b))
                    # y1 = int(y0 + 1000*(a))
                    # x2 = int(x0 - 1000*(-b))
                    # y2 = int(y0 - 1000*(a))
                    # cv2.line(raw_img, (x1,y1), (x2,y2), (0,0,255), line_thickness)
                    num_lines += 1

        lines = np.array(hough).reshape((hough.shape[0], hough.shape[-1]))
        x1_column_index = 0
        left_line_index = np.where(lines[:, x1_column_index] == np.min(lines[:, x1_column_index]))
        left_line = lines[left_line_index][0]
        right_line_index = np.where(lines[:, x1_column_index] == np.max(lines[:, x1_column_index]))
        right_line = lines[right_line_index][0]

        cv2.line(raw_img, (left_line[0],left_line[1]), (left_line[2],left_line[3]), (0,0,255), line_thickness)
        cv2.line(raw_img, (right_line[0],right_line[1]), (right_line[2],right_line[3]), (0,255,0), line_thickness)
        
        # # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
        # lines = calculate_lines(raw_img, hough)
        # # Visualizes the lines
        # # print(lines)
        # lines_visualize = visualize_lines(raw_img, lines)
        # rescale = 0.16
        # new_size = (int(lines_visualize.shape[1] * rescale), int(lines_visualize.shape[0] * rescale)) 
        # rescaled_lines_visualize_output = cv2.resize(lines_visualize, new_size, interpolation=cv2.INTER_AREA)
        # cv2.imshow("Raw Image and Mask", rescaled_lines_visualize_output)
        # cv2.imshow("hough", rescaled_lines_visualize_output)
        # # Overlays lines on raw_img by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
        # raw_img = cv2.addWeighted(raw_img, 0.9, lines_visualize, 1, 1)
                
        # cv2.imshow("Line_Detected_Image", raw_img)
        cv2.waitKey(5)
        # print("Detecting Lines...")

        if DISPLAY_IMAGES:
            three_chan_clean_mask = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
            three_chan_canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
            displayed_img = np.concatenate((raw_img, three_chan_clean_mask), axis=1)
            displayed_img = np.concatenate((displayed_img, three_chan_canny_edges), axis=1)
            display_image("Raw Image and Mask", 0.3, displayed_img)
            
        
        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"): 
            break

cv2.destroyAllWindows()
if __name__ == "__main__":
    init()