import cv2
import pickle
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from insightface.app import FaceAnalysis
import numpy as np

# Configuration
IMAGES_FOLDER = 'images'
MAX_IMAGE_DIMENSION = 800 
OUTPUT_FILE = 'Encoding File.p'

# Initialize face analysis once globally for multiprocessing
app = None

def init_insightface_model():
    global app
    app = FaceAnalysis(name="buffalo_l")  # You can use 'buffalo_l' or 'buffalo_s' (small, faster)
    app.prepare(ctx_id=0, det_size=(640, 640))  # Set GPU (0) or CPU (-1)

def process_single_image(images_folder, filename):
    try:
        filepath = os.path.join(images_folder, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"⚠️ Couldn't read: {filename}")
            return None

        # Resize large images for faster processing
        h, w = img.shape[:2]
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        faces = app.get(img)
        if not faces:
            print(f"🚫 No face in {filename}")
            return None

        # Select the most prominent (largest) face
        largest_face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
        embedding = largest_face.embedding  # 512-d feature vector

        return (embedding, os.path.splitext(filename)[0])

    except Exception as e:
        print(f"🔥 Error processing {filename}: {str(e)}")
        return None

def batch_process_images(image_folder, filenames):
    processor_count = min(cpu_count(), 8)
    with Pool(processes=processor_count, initializer=init_insightface_model) as pool:
        process_task = partial(process_single_image, image_folder)
        results = pool.map(process_task, filenames)

    successful_results = [r for r in results if r is not None]
    if not successful_results:
        return [], []
    
    return zip(*successful_results)

if __name__ == '__main__':
    image_files = os.listdir(IMAGES_FOLDER)
    print(f"🖼️ Found {len(image_files)} images to process")

    encodings, ids = batch_process_images(IMAGES_FOLDER, image_files)

    # Convert encodings (list of numpy arrays) to list of lists for pickling
    encodings = [enc.tolist() if isinstance(enc, np.ndarray) else enc for enc in encodings]

    temp_file = OUTPUT_FILE + '.tmp'
    with open(temp_file, 'wb') as f:
        pickle.dump({'encodings': encodings, 'ids': ids}, f)
    os.replace(temp_file, OUTPUT_FILE)

    success_rate = len(encodings)/len(image_files) * 100
    print(f"✅ Success: {len(encodings)}/{len(image_files)} ({success_rate:.1f}%)")
    print(f"📁 Encodings saved to {OUTPUT_FILE}")
