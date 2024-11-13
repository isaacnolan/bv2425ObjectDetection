from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Google Maps URL with coordinates
url = "https://www.google.com/maps/"

# Load Google Maps
driver.get(url)

# Wait for page to load
time.sleep(5)

# Construct URL to zoom into the specified area (approximation by center point and zoom level)
center_lat = (38.314816 + 38.316639) / 2
center_lng = (-76.548947 + -76.552653) / 2
zoom_level = 18  # Adjust zoom level as needed

# Update URL to focus on the specified area
map_url = f"https://www.google.com/maps/@{center_lat},{center_lng},{zoom_level}z"
driver.get(map_url)
time.sleep(5)

# Take screenshot
screenshot_path = "ImageStitching/ImageDataCollection/map_screenshot.png"
driver.save_screenshot(screenshot_path)
print(f"Screenshot saved to {screenshot_path}")

# Close WebDriver
driver.quit()