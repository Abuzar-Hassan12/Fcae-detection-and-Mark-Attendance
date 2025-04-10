from PIL import Image
import os

images_path = 'images'
output_path = 'resized_images'

os.makedirs(output_path, exist_ok=True)

for img in os.listdir(images_path):
    full_path = os.path.join(images_path, img)
    new_filename = os.path.splitext(img)[0] + ".jpg"
    save_path = os.path.join(output_path, new_filename)

    if os.path.exists(save_path):
        print(f"Skipping {img}: Already resized")
        continue

    try:
        with Image.open(full_path) as FP:
            FP = FP.convert("RGB")
            resized_img = FP.resize((216, 216))
            resized_img.save(save_path, "JPEG")
            print(f"Converted and resized {img} -> {new_filename}")
    
    except Exception as e:
        print(f"Skipping {img}: {e}")
