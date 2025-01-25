#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

const std::string images_path = "../images/googleMaps";
const std::string output_path = "../output/stitchedNew.jpg";
bool crop = true;
bool preprocessing = false;

std::vector<cv::Mat> load_images(const std::string& images_path) {
    std::cout << "[INFO] loading images..." << std::endl;
    std::vector<std::string> image_paths;
    
    // Use std::filesystem to iterate over the directory and collect image file paths
    for (const auto& entry : std::filesystem::directory_iterator(images_path)) {
        if (entry.is_regular_file()) {
            image_paths.push_back(entry.path().string());
        }
    }
    
    // Sort the image paths
    std::sort(image_paths.begin(), image_paths.end());
    
    // Read the images
    std::vector<cv::Mat> images;
    for (const auto& image_path : image_paths) {
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "[WARNING] Could not read image: " << image_path << std::endl;
        } else {
            images.push_back(img);
        }
    }
    return images;
}

std::vector<cv::Mat> resize_images(const std::vector<cv::Mat>& images, int widthThreshold = 1500) {
    std::vector<cv::Mat> resized_images;
    for (const auto& image : images) {
        if (image.cols > widthThreshold) {
            std::cout << "[INFO] Resizing Image..." << std::endl;
            double ratio = static_cast<double>(widthThreshold) / image.cols;
            cv::Size dim(widthThreshold, static_cast<int>(image.rows * ratio));
            cv::Mat resized;
            cv::resize(image, resized, dim);
            resized_images.push_back(resized);
        } else {
            resized_images.push_back(image);
        }
    }
    return resized_images;
}

std::vector<cv::Mat> preprocess_images(const std::vector<cv::Mat>& images) {
    std::vector<cv::Mat> processed_images = images; // Copy input images
    if (preprocessing) {
        std::cout << "[INFO] preprocessing images to improve clarity..." << std::endl;
        for (size_t i = 0; i < processed_images.size(); ++i) {
            // Apply CLAHE to equalize brightness and make sharper
            cv::Mat lab_image;
            cv::cvtColor(processed_images[i], lab_image, cv::COLOR_BGR2Lab);
            std::vector<cv::Mat> lab_planes(3);
            cv::split(lab_image, lab_planes);
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(lab_planes[0], lab_planes[0]);
            cv::merge(lab_planes, lab_image);
            cv::cvtColor(lab_image, processed_images[i], cv::COLOR_Lab2BGR);
        }
    }
    return processed_images;
}

cv::Stitcher::Status stitch_images(const std::vector<cv::Mat>& images, cv::Mat& stitched) {
    std::cout << "[INFO] stitching images..." << std::endl;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
    cv::Stitcher::Status status = stitcher->stitch(images, stitched);
    return status;
}

void crop_image(cv::Mat& stitched) {
    if (crop) {
        std::cout << "[INFO] cropping..." << std::endl;
        // Add a 10px border around the image
        cv::copyMakeBorder(stitched, stitched, 10, 10, 10, 10, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        // Convert to grayscale and threshold
        cv::Mat gray;
        cv::cvtColor(stitched, gray, cv::COLOR_BGR2GRAY);
        cv::Mat thresh;
        cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        // Find contours and crop
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!contours.empty()) {
            // Find the largest contour
            double maxArea = 0;
            int maxIdx = 0;
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = cv::contourArea(contours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    maxIdx = i;
                }
            }
            cv::Rect bounding_rect = cv::boundingRect(contours[maxIdx]);
            stitched = stitched(bounding_rect).clone(); // Clone to ensure data integrity
        }
    }
}

void save_and_display_image(const cv::Mat& stitched, const std::string& output_path) {
    cv::imwrite(output_path, stitched);
    std::cout << "[INFO] stitched image saved to " << output_path << std::endl;
    cv::imshow("Stitched Image", stitched);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::vector<cv::Mat> images = load_images(images_path);
    std::vector<cv::Mat> resized_images = resize_images(images);
    std::vector<cv::Mat> processed_images = preprocess_images(resized_images);
    cv::Mat stitched;
    cv::Stitcher::Status status = stitch_images(processed_images, stitched);

    // Check status of stitching
    if (status == cv::Stitcher::OK) {
        std::cout << "[INFO] image stitching successful!" << std::endl;
        crop_image(stitched);
        save_and_display_image(stitched, output_path);
    } else {
        std::cout << "[ERROR] image stitching failed (error code: " << int(status) << ")" << std::endl;
        if (status == cv::Stitcher::ERR_NEED_MORE_IMGS) {
            std::cout << "[SOLUTION] Need more images to perform stitching." << std::endl;
        } else if (status == cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL) {
            std::cout << "[SOLUTION] Homography estimation failed. Try improving image overlap or reducing blur." << std::endl;
        } else if (status == cv::Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL) {
            std::cout << "[SOLUTION] Camera parameter adjustment failed. Try changing image capture settings." << std::endl;
        } else {
            std::cout << "[SOLUTION] Unknown error." << std::endl;
        }
    }
    return 0;
}
