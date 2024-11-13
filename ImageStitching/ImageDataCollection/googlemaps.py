import requests
from PIL import Image
from io import BytesIO

# Google Maps Static API key
api_key = 'insertkey'

image_path = 'ImageStitching/ImageDataCollection/map_screenshot.png'

# Coordinates for the area
southwest = "38.314816,-76.552653"  # bottom-left corner
northeast = "38.316639,-76.548947"  # top-right corner

# Calculate the center point from these coordinates
center_lat = (38.314816 + 38.316639) / 2
center_lng = (-76.548947 + -76.552653) / 2
center = f"{center_lat},{center_lng}"

# Define zoom level and image dimensions (can adjust for higher resolution)
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
    image.save("satellite_map.png")
    print("Satellite map image saved as satellite_map.png")
else:
    print("Failed to retrieve map:", response.status_code, response.text)
