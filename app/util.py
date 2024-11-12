from PIL import Image

# Load your high-resolution image
high_res_image_path = "lowresimg.png"
img = Image.open(high_res_image_path)

# Resize to create a low-res version
low_res_img = img.resize((128, 128), Image.BILINEAR)  # Adjust size as needed

# Save or use the low-res image
low_res_img.save("low_res_image.png")
