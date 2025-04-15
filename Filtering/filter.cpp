#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include "yolov8.h"

using namespace std;

unordered_map<string, int> filter_objects(const vector<string>& images, const string& model_path, int min_occurrences = 3) {
    /*
    Filters out objects detected in less than `min_occurrences` frames.
    
    :param images: List of image file paths.
    :param model_path: Path to the YOLO model.
    :param min_occurrences: Minimum number of times an object must appear to be retained.
    :return: Map of object labels with occurrences >= min_occurrences.
    */

    // Load YOLO model
    YOLOv8 model(model_path);
    unordered_map<string, int> object_counts;

    for (const auto& image : images) {
        // Run YOLO model on image
        auto results = model.detect(image);
        for (const auto& result : results) {
            // Get label name
            string label = result.label;
            object_counts[label]++;
        }
    }

    // Filter out objects with fewer than min_occurrences
    unordered_map<string, int> filtered_objects;
    for (const auto& [label, count] : object_counts) {
        if (count >= min_occurrences) {
            filtered_objects[label] = count;
        }
    }

    return filtered_objects;
}

int main() {
    vector<string> images = {"image1.jpg", "image2.jpg", "image3.jpg"};
    string model_path = "yolov8.pt";  // Replace with model path
    unordered_map<string, int> filtered_results = filter_objects(images, model_path);
    
    for (const auto& [label, count] : filtered_results) {
        cout << label << ": " << count << endl;
    }

    return 0;
}
