import numpy as np
from collections import defaultdict

def filter_detected_objects(points3d, threshold=1.0):
    """
    Filters detected objects based on proximity and occurrence in multiple images.

    Args:
        points3d (np.ndarray): Array of 3D points (shape = Nx3) from Allen's function.
        threshold (float): Distance threshold to consider points as the same object.

    Returns:
        dict: Filtered hash map of objects with their coordinates and occurrence counts.
    """
    # Hash map to store unique objects
    object_map = defaultdict(lambda: {"coordinates": [], "count": 0})

    for point in points3d:
        x, y, z = point

        # Check if the point is "close enough" to an existing object
        found_match = False
        for obj_id, obj_data in object_map.items():
            existing_coords = obj_data["coordinates"]
            if np.linalg.norm(np.array(existing_coords) - np.array([x, y, z])) <= threshold:
                # Update the existing object's count and average coordinates
                obj_data["coordinates"] = [
                    (existing_coords[0] * obj_data["count"] + x) / (obj_data["count"] + 1),
                    (existing_coords[1] * obj_data["count"] + y) / (obj_data["count"] + 1),
                    (existing_coords[2] * obj_data["count"] + z) / (obj_data["count"] + 1),
                ]
                obj_data["count"] += 1
                found_match = True
                break

        if not found_match:
            # Add a new object to the map
            obj_id = len(object_map)
            object_map[obj_id]["coordinates"] = [x, y, z]
            object_map[obj_id]["count"] = 1

    # Filter out objects that appear fewer than 3 times
    filtered_map = {
        obj_id: obj_data
        for obj_id, obj_data in object_map.items()
        if obj_data["count"] >= 3
    }

    return filtered_map

if __name__ == "__main__":
    # Example 3D points from triangulation function
    points3d_example = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [5.0, 6.0, 7.0],
        [1.2, 2.2, 3.2],
        [5.1, 6.1, 7.1],
        [8.0, 9.0, 10.0],
    ])

    # Filter objects
    filtered_objects = filter_detected_objects(points3d_example, threshold=0.5)

    # Print results
    print("Filtered objects:")
    for obj_id, obj_data in filtered_objects.items():
        print(f"Object {obj_id}: Coordinates = {obj_data['coordinates']}, Count = {obj_data['count']}")
