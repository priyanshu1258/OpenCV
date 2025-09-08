import cv2
import os
import numpy as np
from PIL import Image

# Paths
DATASET_PATH = "dataset"
MODEL_FILE = "trainer.yml"
LABEL_FILE = "labels.npy"

# Ensure dataset folder exists
os.makedirs(DATASET_PATH, exist_ok=True)

# --------------------- ADD NEW USER ---------------------
def add_user(user_name):
    user_path = os.path.join(DATASET_PATH, user_name)
    os.makedirs(user_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            file_name = os.path.join(user_path, f"{count}.jpg")
            cv2.imwrite(file_name, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Image {count}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Creating Dataset", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Dataset created for {user_name} at {user_path}")


# --------------------- TRAIN MODEL ---------------------
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []
    label_map = {}
    current_id = 0

    for user_name in os.listdir(DATASET_PATH):
        user_path = os.path.join(DATASET_PATH, user_name)
        if not os.path.isdir(user_path):
            continue

        label_map[current_id] = user_name
        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, 'uint8')
            face_samples.append(img_np)
            ids.append(current_id)
        current_id += 1

    if len(face_samples) == 0:
        print("⚠️ No data found. Please add a user first.")
        return

    recognizer.train(face_samples, np.array(ids))
    recognizer.save(MODEL_FILE)
    np.save(LABEL_FILE, label_map)
    print("✅ Training complete. Model saved.")


# --------------------- RECOGNIZE FACES ---------------------
def recognize_faces():
    if not os.path.exists(MODEL_FILE):
        print("⚠️ Model not trained yet. Train it first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    label_map = np.load(LABEL_FILE, allow_pickle=True).item()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)

            if conf < 70:
                name = label_map[id_]
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow("Face Recognition", frame)

        # ✅ Allow exit with 'q' or ESC key or closing window
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 = ESC
            break
        if cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()



# --------------------- MAIN MENU ---------------------
if __name__ == "__main__":
    while True:
        print("\n=== Face Recognition System ===")
        print("1. Add New User")
        print("2. Train Model")
        print("3. Run Face Recognition")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            name = input("Enter user name: ")
            add_user(name)
        elif choice == "2":
            train_model()
        elif choice == "3":
            recognize_faces()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice, try again.")

