from PIL import Image
import os

def overlay_images(input_folder, output_image_path, transparency=0):
    # List all images in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    
    if not image_files:
        print("No images found in the specified folder.")
        return
    
    # Open the first image to get dimensions and use as the base
    base_image = Image.open(os.path.join(input_folder, image_files[0])).convert("RGBA")
    base_image = base_image.copy()  # Make a copy to avoid modifying the original
    width, height = base_image.size

    # Convert images to the same size and overlay them
    for image_file in image_files[1:]:
        overlay_image = Image.open(os.path.join(input_folder, image_file)).convert("RGBA")
        
        # Resize overlay image to match base dimensions, if needed
        overlay_image = overlay_image.resize((width, height))
        
        # Apply transparency to overlay image
        overlay_with_transparency = overlay_image.copy()
        overlay_with_transparency.putalpha(int(255 * transparency))
        
        # Composite the images
        base_image = Image.alpha_composite(base_image, overlay_with_transparency)

    # Convert final image to RGB and save
    final_image = base_image.convert("RGB")
    final_image.save(output_image_path, format="JPEG")
    print(f"Overlay image saved as {output_image_path}")

# Example usage
input_folder = 'output_images'  # Folder with individual grid images
output_image_path = 'overlay_result.jpg'  # Output image file
overlay_images(input_folder, output_image_path, transparency=0.2)