from PIL import Image
import numpy as np

def check_alpha_impact(image_path):
    # Open the image
    img = Image.open(image_path).convert("RGBA")
    
    # Split the image into R, G, B, and Alpha channels
    r, g, b, alpha = img.split()
    
    # Convert the alpha channel to a NumPy array
    alpha_array = np.array(alpha)
    
    # Check if all alpha values are 255 (fully opaque)
    if np.all(alpha_array == 255):
        return "Alpha channel has no impact (fully opaque)."
    
    # Check if all alpha values are 0 (fully transparent)
    elif np.all(alpha_array == 0):
        return "Alpha channel has no impact (fully transparent)."
    
    # If alpha values vary
    else:
        unique_alpha_values = np.unique(alpha_array)
        return f"Alpha channel has impact. Unique alpha values: {unique_alpha_values}"

# Example usage
image_path = "/Users/guoyuzhang/University/Y5/diss/code/data/CLEVR_v1.0/images/test/CLEVR_test_000000.png"  # Replace with your image path
print(check_alpha_impact(image_path))

