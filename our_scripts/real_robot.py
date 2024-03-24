import cv2
import numpy as np

class RealRobotics():

    def __init__(self):
        self.goal_complete = False

    def reset(self):
        self.goal_complete = False


    # Function to identify the red star based on its color and shape
    def identify_red_star(self, hsv):
        # Define range of red color in HSV
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # Second range to catch red
        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        # Combine the masks in case red is across the boundary in the HSV space
        red_mask = mask1 + mask2

        # Find contours in the red mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to store the identified star contour and its position
        star_contour = None
        star_position = None

        # Iterate over contours to find the one with the most vertices (which should be the star)
        for cnt in contours:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

            # Star should have more vertices than the cylinder; this is a simplistic check that may need refinement
            if len(approx) > 6:  # Assuming the star has more than 6 vertices
                star_contour = cnt
                # Calculate the centroid of the contour for position
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    star_position = (cx, cy)
                break

        return star_position, star_contour

    # Function to define the goal area on the board
    def define_goal_area(self, image):
        # Define the goal area as a 50% square of the image's width and height from the bottom right corner
        goal_area_width = int(image.shape[1] * 0.5)
        goal_area_height = int(image.shape[0] * 0.5)
        goal_area_top_left = (image.shape[1] - goal_area_width, image.shape[0] - goal_area_height)
        goal_area_bottom_right = (image.shape[1], image.shape[0])

        return goal_area_top_left, goal_area_bottom_right

    def check_goal(self, hsv, image):
        # Identify the red star
        star_position, star_contour = self.identify_red_star(hsv)

        # Define the goal area
        goal_area_top_left, goal_area_bottom_right = self.define_goal_area(image)

        if star_position:
            cv2.drawContours(image, [star_contour], -1, (0, 255, 0), 2)
            in_goal_area = goal_area_top_left[0] <= star_position[0] <= goal_area_bottom_right[0] and goal_area_top_left[1] <= star_position[1] <= goal_area_bottom_right[1]
        else:
            in_goal_area = False
        
        return in_goal_area
    
    def reward(self, image_path):
        image = cv2.imread(image_path)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        reward_out = 0.
         
        goal_completed = self.check_goal(hsv, image)

        if goal_completed and self.goal_complete == False:
            self.goal_complete = True
            reward_out = 1.
        
        return reward_out

