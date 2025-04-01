import cv2
import face_recognition
import pickle
import os

# Define the path to the folder containing student images
images_Path = 'images'
stud_images_path = os.listdir(images_Path)

# Initialize lists to store encodings and corresponding IDs
encoded_list = []
stud_ID = []

def find_encodings(images_path_list):
    valid_encodings = []
    valid_ids = []
    
    for path in images_path_list:
        full_path = os.path.join(images_Path, path)
        # Load image
        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            continue
        
        # Convert image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(img_rgb, model='hog')  # 'cnn' for better accuracy
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        if len(face_encodings) == 0:
            print(f"No face detected in {path}. Skipping.")
            continue
        
        # Select the largest face (best candidate for ID matching)
        largest_face_index = max(range(len(face_locations)), key=lambda i: (face_locations[i][2] - face_locations[i][0]) * (face_locations[i][1] - face_locations[i][3]))
        valid_encodings.append(face_encodings[largest_face_index])
        
        # Add the corresponding student ID (filename without extension)
        valid_ids.append(os.path.splitext(path)[0])
    
    return valid_encodings, valid_ids

# Process all images and get valid encodings with IDs
encoded_list, stud_ID = find_encodings(stud_images_path)

# Save the encodings with corresponding IDs to a temporary file first
temp_file = 'Encoding File.temp.p'
with open(temp_file, 'wb') as encoded_file:
    pickle.dump([encoded_list, stud_ID], encoded_file)

# Rename temp file to avoid corruption issues
os.replace(temp_file, 'Encoding File.p')

print(f"Successfully encoded {len(encoded_list)} images out of {len(stud_images_path)}")
