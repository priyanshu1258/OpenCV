import cv2
import os

def save_face(image, faces, folder="dataset"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    i = 0
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        file_path = os.path.join(folder, f"face_{i}.jpg")
        cv2.imwrite(file_path, face)
        i += 1
    print(f"{i} face(s) saved in {folder}/")
