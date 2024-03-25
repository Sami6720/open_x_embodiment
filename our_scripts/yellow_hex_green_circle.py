import cv2
import numpy as np

class YellowHexGreenHeart():

    def __init__(self):
        self.done = False
        self.initial_yellow_heart_location = None
        self.initial_green_circle_location = None
        self.first = True
        
    def reset(self):
        self.done = False
        self.initial_yellow_heart_location = None
        self.initial_green_circle_location = None
        self.first = True

    # Function to identify the yellow heart
    def find_yellow_heart(self, image):
        
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for yellow color and create mask
        # Adjusting the range for yellow to be more inclusive
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for the heart shape based on the assumption that it has a smooth contour
        heart_contours = []
        for contour in contours:
            # Approximate the contour to smooth it out
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Assume the heart has at least 5 vertices (after approximation) and area larger than a threshold to exclude noise
            if len(approx) >= 5 and cv2.contourArea(contour) > 200:
                heart_contours.append(contour)
        
        # If at least one heart-shaped contour is found, return True and the location of the first one
        if len(heart_contours) > 0:
            x, y, w, h = cv2.boundingRect(heart_contours[0])
            location = (x + w//2, y + h//2)  # Center of the bounding box
            return True, location
        
        return False, None

    def find_green_circle(self, image):
        
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for green color and create mask
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for the circle shape based on the assumption that it will be a smooth, round contour
        circle_contours = []
        for contour in contours:
            # Approximate the contour to smooth it out
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # The circle will have a more rounded shape, so we look for contours with many vertices
            # Check if the aspect ratio is approximately 1 to ensure it's round
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if len(approx) > 8 and 0.8 < aspect_ratio < 1.2:
                circle_contours.append(contour)
        
        # If at least one round-shaped contour is found, return True and the location of the first one
        if len(circle_contours) > 0:
            x, y, w, h = cv2.boundingRect(circle_contours[0])
            location = (x + w//2, y + h//2)  # Center of the bounding box
            return True, location
        
        return False, None
    def get_object_locations(self, image):
        # Find the yellow heart
        yellow_heart_found, yellow_heart_location = self.find_yellow_heart(image)
        # Find the green circle
        green_circle_found, green_circle_location = self.find_green_circle(image)
    
        return yellow_heart_found, yellow_heart_location, green_circle_found, green_circle_location



    def check_goal_completion_with_adjusted_proximity(self, initial_yellow_heart_location, initial_green_circle_location, current_image):
        # Adjust the proximity threshold (in pixels) to better match the scale of the image
        proximity_threshold = 100  # Increased threshold for proximity check
        
        # Get the current locations
        current_yellow_heart_found, current_yellow_heart_location, current_green_circle_found, current_green_circle_location = self.get_object_locations(current_image)
        
        # Check if both objects were found in the current state
        if not current_yellow_heart_found or not current_green_circle_found:
            return False

        # Check if the yellow heart has moved to the right of the green circle from its initial position
        moved_to_right = current_yellow_heart_location[0] > initial_green_circle_location[0]
        
        # Check the proximity on the x-axis (horizontal distance)
        is_close = abs(current_yellow_heart_location[0] - initial_green_circle_location[0]) <= proximity_threshold

        # The goal is achieved if the yellow heart has moved to the right and is close to the green circle
        return moved_to_right and is_close
    
    def checker(self, image):
        if self.first == True:
            _, self.initial_yellow_heart_location, _, self.initial_green_circle_location = self.get_object_locations(image)
            self.first = False
        

        done = self.check_goal_completion_with_adjusted_proximity(self.initial_yellow_heart_location, self. initial_green_circle_location, image)

        return done