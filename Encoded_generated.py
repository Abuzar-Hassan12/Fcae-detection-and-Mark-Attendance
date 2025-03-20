import cv2
import face_recognition
import pickle
import os

# Define the path to the folder where images are stored
images_Path = 'resized_images'
encoding_file_path = 'Encoding_File.p'

# Load existing encodings if the file exists
if os.path.exists(encoding_file_path):
    with open(encoding_file_path, 'rb') as f:
        founded_encodings_with_ID = pickle.load(f)
    existing_encodings, existing_IDs = founded_encodings_with_ID
else:
    existing_encodings, existing_IDs = [], []

# List all the image files in the directory
stud_images_path = os.listdir(images_Path)

# Initialize lists for new images
stud_img_List = []
stud_ID = []

# Load new images that have not been encoded before
for path in stud_images_path:
    student_id = os.path.splitext(path)[0]
    
    if student_id in existing_IDs:
        print(f"Skipping already encoded image: {path}")
        continue  # Skip already processed images
    
    img_path = os.path.join(images_Path, path)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Error: Could not load image {path}. Check if the file exists and is accessible.")
        continue
    
    stud_img_List.append(img)
    stud_ID.append(student_id)

# Function to find encodings
def findencoding(stud_img_List, stud_ID):
    encoded_list = []
    valid_IDs = []
    
    for i, img in enumerate(stud_img_List):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        
        if len(encodings) == 1:  # Ensure only one face is detected
            encoded_list.append(encodings[0])
            valid_IDs.append(stud_ID[i])
        elif len(encodings) > 1:
            print(f"⚠️ Warning: Multiple faces detected in {stud_ID[i]}, skipping it!")
        else:
            print(f"⚠️ Warning: No face detected in {stud_ID[i]}, skipping it!")
    
    return encoded_list, valid_IDs

# Find encodings for new images
if stud_img_List:
    new_encodings, new_IDs = findencoding(stud_img_List, stud_ID)
    
    # Append new encodings to existing ones
    existing_encodings.extend(new_encodings)
    existing_IDs.extend(new_IDs)
    
    # Save updated encodings
    with open(encoding_file_path, 'wb') as f:
        pickle.dump([existing_encodings, existing_IDs], f)
    
    print("✅ Encodings updated successfully!")
else:
    print("✅ No new images to encode.")
