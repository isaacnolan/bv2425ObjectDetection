import requests
from PIL import Image
from io import BytesIO

# Google Maps Static API key
api_key = 'key'

# Coordinates for the area
southwest = "38.314816,-76.552653"  # bottom-left corner
northeast = "38.316639,-76.548947"  # top-right corner

# Calculate the center point from these coordinates
center_lat = (38.314816 + 38.316639) / 2
center_lng = (-76.548947 + -76.552653) / 2
center = f"{center_lat},{center_lng}"

# Define zoom level and image dimensions (adjust for higher resolution if needed)
zoom = 19  # Adjust as needed for closer or wider view
width, height = 600, 600  # Adjust image dimensions

# Google Static Maps API URL
url = f"https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&size={width}x{height}&maptype=satellite&key={api_key}"

# Send request to the API
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    # Open the image from the response
    image = Image.open(BytesIO(response.content))
    image.save("ImageStitching/ImageDataCollection/map_screenshot.png")
    print("Satellite map image saved as map_screenshot.png")

    # Define the rotation angle to align the road horizontally
    # This angle is assumed; adjust as needed for the specific image.
    rotation_angle = 14  # Replace with the actual angle

    # Rotate the image
    rotated_image = image.rotate(rotation_angle, expand=True)

    # Define the overlap ratio (20-30% overlap, we use 25% here)
    overlap_ratio = 0.25

    # New width and height of the rotated image
    rotated_width, rotated_height = rotated_image.size

    # Calculate the width for each section with overlap
    section_width = int(rotated_width / 3 * (1 + overlap_ratio))

    # Generate and save the three overlapping horizontal sections
    for i in range(3):
        left = int(i * (rotated_width / 3) * (1 - overlap_ratio))
        right = left + section_width

        # Crop the rotated image for this section
        section = rotated_image.crop((left, 0, min(right, rotated_width), rotated_height))
        section.save(f"ImageStitching/ImageDataCollection/map_section_{i+1}.png")
        print(f"Saved rotated section {i+1} with overlap as map_section_{i+1}.png")
else:
    print("Failed to retrieve map:", response.status_code, response.text)
