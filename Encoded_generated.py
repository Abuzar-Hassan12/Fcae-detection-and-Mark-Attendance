import cv2
import face_recognition
import pickle
import os


#NOTE importing the Students images

# Define the path to the folder where modes (images) are stored
images_Path = 'images'

# List all the contents in the specified 'modeFolderPath'
stud_images_path = os.listdir(images_Path)

# Initialize an empty list to store images that will be loaded
stud_img_List = []
stud_ID=[]
# Iterate through each path (image) in the directory
for path in stud_images_path:
    # Join the folder path with the current file/folder name to get the full path
    # Read the image from the full path using OpenCV and append it to the list
    stud_img_List.append(cv2.imread(os.path.join(images_Path, path)))
    #ID are like 4313.png ->split -> '4313','.png'
    #we need actually 4313 (that is at 0 index) split text
    stud_ID.append(os.path.splitext(path)[0])
    
# NOTE print(stud_ID) # splited IDs of all Student

# NOTE find Encodings of the images of students
def findencoding(stud_img_List):
    encoded_list=[]
    for img in stud_img_List:
        # FOR encoding first step is to change color
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # 2nd find Encodings
        encode = face_recognition.face_encodings(img)[0]
        encoded_list.append(encode)
    
    return encoded_list

#Encoded images are now founded 
founded_encodings=findencoding(stud_img_List)

# we need file where we can store encodings with relevents ID so we can use that file 
# in WebCam
founded_encodings_with_ID =[founded_encodings,stud_ID]

encoded_file =open('Encoding File.p','wb')
pickle.dump(founded_encodings_with_ID,encoded_file)
encoded_file.close()
