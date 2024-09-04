import cv2
import numpy as np
import os
import random

# Set up the window size
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 640

# Set the minimum and maximum sizes for images
MIN_IMAGE_SIZE = (20, 20)
MAX_IMAGE_SIZE = (50, 50)

# Helper function to check for overlap between rectangles
def check_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

# Load character images and labels
character_images = []
character_labels = []
character_dir = "data\OCR\Train\images"
label_dir = "data\OCR\Train\labels"

for filename in os.listdir(character_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(character_dir, filename)
        character_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        character_images.append(character_image)

        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        with open(label_path, 'r') as file:
            data = file.read().splitlines()
            label_id = int(data[0].split()[0])  # Extract the first number (label ID)
        character_labels.append(label_id)

# Load shape images
shape_images = []
shape_dir = "outputShape"
for filename in os.listdir(shape_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(shape_dir, filename)
        shape_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        shape_images.append(shape_image)

# Load background images
background_images = []
background_dir = "backgroundData"
for filename in os.listdir(background_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(background_dir, filename)
        background_image = cv2.imread(image_path)
        background_image = cv2.resize(background_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
        background_images.append(background_image)

# Create directories to save the images and labels
output_dir = "output"
label_output_dir = "labels"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(label_output_dir):
    os.makedirs(label_output_dir)

# Calculate the number of output images needed
num_output_images = (len(character_images) + 9) // 10

# Generate synthetic images and labels
image_count = 0
while image_count < num_output_images:
    background_image = random.choice(background_images)
    canvas = background_image.copy()
    placed_images = []
    placed_labels = []

    for _ in range(10):
        if not character_images:
            break

        character_image = character_images.pop(0)
        character_label = character_labels.pop(0)
        shape_image = random.choice(shape_images)

        # Choose a random size for the character and shape images
        image_width = random.randint(MIN_IMAGE_SIZE[0], MAX_IMAGE_SIZE[0])
        image_height = random.randint(MIN_IMAGE_SIZE[1], MAX_IMAGE_SIZE[1])

        # Resize the character image
        resized_character_image = cv2.resize(character_image, (image_width, image_height))

        # Check the number of channels in the resized character image
        num_channels = resized_character_image.shape[2] if len(resized_character_image.shape) > 2 else 1

        # Convert the character image to a 3-channel image
        if num_channels == 1:
            resized_character_image = cv2.cvtColor(resized_character_image, cv2.COLOR_GRAY2BGR)
        elif num_channels == 3:
            # No need for conversion, the image is already in BGR format
            pass
        else:
            print(f"Warning: Unsupported number of channels ({num_channels}) in the character image.")
            continue

        # Resize the shape image to be 20% larger than the character image
        shape_width = int(image_width * 2)
        shape_height = int(image_height * 2)
        resized_shape_image = cv2.resize(shape_image, (shape_width, shape_height))

        # Find a random position that doesn't overlap with existing images
        overlap = True
        max_attempts = 1000
        attempts = 0
        while overlap and attempts < max_attempts:
            x = random.randint(0, WINDOW_WIDTH - resized_character_image.shape[1])
            y = random.randint(0, WINDOW_HEIGHT - resized_character_image.shape[0])
            character_rect = (x, y, resized_character_image.shape[1], resized_character_image.shape[0])
            shape_rect = (x - (resized_shape_image.shape[1] - resized_character_image.shape[1]) // 2,
                          y - (resized_shape_image.shape[0] - resized_character_image.shape[0]) // 2,
                          resized_shape_image.shape[1], resized_shape_image.shape[0])
            overlap = False

            for placed_rect in placed_images:
                if check_overlap(character_rect, placed_rect) or check_overlap(shape_rect, placed_rect):
                    overlap = True
                    break

            attempts += 1

        if attempts >= max_attempts:
            print(f"Warning: Could not place all images on canvas for output image {image_count + 1}")
            break

        # Place the resized shape image on the canvas
        shape_x = max(0, min(x - (resized_shape_image.shape[1] - resized_character_image.shape[1]) // 2, WINDOW_WIDTH - resized_shape_image.shape[1]))
        shape_y = max(0, min(y - (resized_shape_image.shape[0] - resized_character_image.shape[0]) // 2, WINDOW_HEIGHT - resized_shape_image.shape[0]))
        end_x = min(shape_x + resized_shape_image.shape[1], WINDOW_WIDTH)
        end_y = min(shape_y + resized_shape_image.shape[0], WINDOW_HEIGHT)

        canvas[shape_y:end_y, shape_x:end_x] = resized_shape_image[0:end_y - shape_y, 0:end_x - shape_x]

        # Place the resized character image on top of the shape image
        canvas[y:y+resized_character_image.shape[0], x:x+resized_character_image.shape[1]] = resized_character_image

        # Add the character rectangle and its label to the list of placed images and labels
        placed_images.append(character_rect)
        placed_labels.append(character_label)

    # Save the synthetic image
    image_count += 1
    output_path = os.path.join(output_dir, f"output_{image_count}.png")
    label_output_path = os.path.join(label_output_dir, f"output_{image_count}.txt")
    cv2.imwrite(output_path, canvas)

    # Save the labels for the current image
    with open(label_output_path, "w") as f:
        for rect, label_id in zip(placed_images, placed_labels):
            x_center = (rect[0] + rect[2] / 2) / WINDOW_WIDTH
            y_center = (rect[1] + rect[3] / 2) / WINDOW_HEIGHT
            width = rect[2] / WINDOW_WIDTH
            height = rect[3] / WINDOW_HEIGHT
            label_data = f"{label_id} {x_center} {y_center} {width} {height}\n"
            f.write(label_data)