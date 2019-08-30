import face_recognition
from PIL import Image
from imutils import paths
import os
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("data"))

if not os.path.exists(str("face_images")):
    os.makedirs("face_images")
data = []
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	print(imagePath)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	print('after rgb')
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	print('after boxes')
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
	print('after encodings')
	# build a dictionary of the image path, bounding box location,
	# and facial encodings for the current image
	d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
		for (box, enc) in zip(boxes, encodings)]
	data.extend(d)
	print('after data extend')
    # image = face_recognition.load_image_file(imagePath)
    # face_locations = face_recognition.face_locations(image)

    # img_source = Image.open(imagePath)
    # # print(face_locations)
    # for j, (top, right, bottom, left) in enumerate(face_locations):
    #     print(j, (top, right, bottom, left))
    #     crop_img = img_source.crop((left,top,right,bottom))
    #     crop_img.save("face_images/" + str(i) + "_" + str(j) + ".jpg")
print("[INFO] serializing encodings...")
f = open("encodings.pickle2", "wb")
f.write(pickle.dumps(data))
f.close()
