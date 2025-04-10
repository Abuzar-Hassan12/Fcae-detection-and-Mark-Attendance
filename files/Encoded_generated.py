import cv2
import face_recognition
import pickle
import os
from multiprocessing import Pool, cpu_count
from functools import partial

# Configuration
IMAGES_FOLDER = 'images'
MAX_IMAGE_DIMENSION = 800 
FACE_DETECTION_MODEL = 'hog'  # Faster than 'cnn'
NUM_ENCODING_JITTERS = 1  # Keep low for speed
OUTPUT_FILE = 'Encoding File.p'

def process_single_image(images_folder, filename):
    try:
        # Load image
        filepath = os.path.join(images_folder, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"⚠️ Couldn't read: {filename}")
            return None

        # Optimize image processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize large images while maintaining aspect ratio
        h, w = img_rgb.shape[:2]
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        if scale < 1:
            img_rgb = cv2.resize(img_rgb, 
                               (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        # Face detection with optimized model
        face_locations = face_recognition.face_locations(img_rgb, model=FACE_DETECTION_MODEL)
        if not face_locations:
            print(f"🚫 No face in {filename}")
            return None

        # Select largest face (most prominent)
        largest_face = max(face_locations, 
                          key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
        
        # Get encoding for only the largest face
        encoding = face_recognition.face_encodings(
            img_rgb, 
            known_face_locations=[largest_face],
            num_jitters=NUM_ENCODING_JITTERS
        )[0]  # Single face processed

        return (encoding, os.path.splitext(filename)[0])

    except Exception as e:
        print(f"🔥 Error processing {filename}: {str(e)}")
        return None

def batch_process_images(image_folder, filenames):
    # Smart parallel processing with CPU core utilization
    processor_count = min(cpu_count(), 8)  # Prevent overloading
    process_task = partial(process_single_image, image_folder)

    with Pool(processes=processor_count) as pool:
        results = pool.map(process_task, filenames)

    # Filter and unpack successful results
    successful_results = [r for r in results if r is not None]
    if not successful_results:
        return [], []
    
    return zip(*successful_results)  # Unzips into (encodings, ids)

if __name__ == '__main__':
    image_files = os.listdir(IMAGES_FOLDER)
    print(f"🖼️ Found {len(image_files)} images to process")

    encodings, ids = batch_process_images(IMAGES_FOLDER, image_files)

    # Safe file writing with atomic replace
    temp_file = OUTPUT_FILE + '.tmp'
    with open(temp_file, 'wb') as f:
        pickle.dump({'encodings': encodings, 'ids': ids}, f)
    os.replace(temp_file, OUTPUT_FILE)

    success_rate = len(encodings)/len(image_files) * 100
    print(f"✅ Success: {len(encodings)}/{len(image_files)} ({success_rate:.1f}%)")
    print(f"📁 Encodings saved to {OUTPUT_FILE}")
