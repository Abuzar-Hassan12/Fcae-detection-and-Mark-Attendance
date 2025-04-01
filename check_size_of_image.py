from PIL import Image
import os
imageS_path= 'images'
for img in os.listdir(imageS_path):
   full_path=os.path.join(imageS_path,img) 
   with Image.open(full_path) as FP: # open imge and FP.size Find size of an imagee
      print(img,FP.size)