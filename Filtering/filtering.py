import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean

# Placeholder function that gets the real-world location of an object given its pixel location
# In practice, this would be replaced by your actual function
def get_object_location_from_pixel(pixel_coords):
    # Dummy function, assume a 1:1 mapping for simplicity
    return np.array(pixel_coords)

class ObjectTracker:
    def __init__(self, threshold=3, proximity_threshold=10):
        self.detected_objects = defaultdict(list)  # Track objects across multiple snapshots
        self.threshold = threshold  # Minimum detections required to confirm object
        self.proximity_threshold = proximity_threshold  # How close consecutive detections should be considered the same object

    def update(self, objects_in_frame):
        """
        Updates the object tracker with the detected objects in the current frame.
        
        :param objects_in_frame: List of tuples, where each tuple contains pixel coordinates (x, y) of an object
        """
        for pixel_coords in objects_in_frame:
            object_location = get_object_location_from_pixel(pixel_coords)
            self._register_object(object_location)

    def _register_object(self, object_location):
        """
        Register an object location, either adding it to an existing object track
        or creating a new one if no close matches are found.
        """
        found = False
        # Check if the object location is near any existing object
        for tracked_object, locations in self.detected_objects.items():
            if locations:
                # Check the distance to the most recent location of the tracked object
                last_known_location = locations[-1]
                if euclidean(object_location, last_known_location) < self.proximity_threshold:
                    self.detected_objects[tracked_object].append(object_location)
                    found = True
                    break
        
        if not found:
            # If no match was found, register this as a new object
            self.detected_objects[len(self.detected_objects)] = [object_location]

    def get_confirmed_objects(self):
        """
        Returns the list of confirmed objects that have been detected 
        at least `threshold` times across multiple snapshots.
        """
        confirmed_objects = []
        for tracked_object, locations in self.detected_objects.items():
            if len(locations) >= self.threshold:
                confirmed_objects.append(locations[-1])  # Get the most recent location of the confirmed object
        return confirmed_objects

# Example usage
if __name__ == "__main__":
    tracker = ObjectTracker(threshold=3, proximity_threshold=10)
    
    # Simulate multiple frames of detections
    frames = [
        [(100, 150), (200, 250)],   # Frame 1: two objects detected
        [(105, 155), (195, 245)],   # Frame 2: slightly shifted, but same two objects
        [(110, 160), (205, 255)],   # Frame 3: again, slight shift in position
        [(300, 400)]                # Frame 4: new object far from previous ones
    ]
    
    # Update the tracker with objects detected in each frame
    for frame in frames:
        tracker.update(frame)
    
    # Get confirmed objects
    confirmed_objects = tracker.get_confirmed_objects()
    
    print("Confirmed objects:", confirmed_objects)
