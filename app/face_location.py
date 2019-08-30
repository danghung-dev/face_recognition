import face_recognition
from PIL import Image
from imutils import paths
import os

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("data"))

if not os.path.exists(str("face_images")):
    os.makedirs("face_images")
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    image = face_recognition.load_image_file(imagePath)
    face_locations = face_recognition.face_locations(image)

    img_source = Image.open(imagePath)
    # print(face_locations)
    for j, (top, right, bottom, left) in enumerate(face_locations):
        print(j, (top, right, bottom, left))
        crop_img = img_source.crop((left,top,right,bottom))
        crop_img.save("face_images/" + str(i) + "_" + str(j) + ".jpg")
