from PIL import Image
import os

def split_image_with_overlap(image_path, output_folder, grid_size=(3, 3), overlap=(50, 50)):
    # Open the image
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Calculate size of each grid cell with overlap
    x_cells, y_cells = grid_size
    cell_width = (img_width // x_cells)
    cell_height = (img_height // y_cells)
    overlap_x, overlap_y = overlap

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate the grid of images
    for i in range(x_cells):
        for j in range(y_cells):
            # Calculate the coordinates for cropping with overlap
            left = max(i * cell_width - overlap_x, 0)
            upper = max(j * cell_height - overlap_y, 0)
            right = min(left + cell_width + overlap_x * 2, img_width)
            lower = min(upper + cell_height + overlap_y * 2, img_height)
            
            # Crop the image and save with filename format x_y.jpg
            cropped_img = image.crop((left, upper, right, lower))
            cropped_img.save(os.path.join(output_folder, f"{i}_{j}.jpg"), format="JPEG")

    print("Image split into grid with overlap successfully.")

# Example usage
image_path = 'images/yosemite.jpg'
output_folder = 'output_images'
split_image_with_overlap(image_path, output_folder, grid_size=(3, 3), overlap=(50, 50))
