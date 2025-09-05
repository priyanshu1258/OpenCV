# Face Recognition Project using OpenCV  

## 📌 Overview  
This project implements a **Face Recognition System** using **OpenCV**.  
The system can:  
- Detect faces in images and videos.  
- Recognize and identify individuals based on a trained dataset.  
- Support real-time recognition using a webcam.  

---

## ⚙️ Features  
- Face detection using **Haar Cascade Classifier** / **DNN**.  
- Face recognition using **LBPH (Local Binary Patterns Histogram)**.  
- Real-time recognition via webcam.  
- Easy to extend by adding more training images.  

---

## 🛠️ Tech Stack  
- **Language**: Python 3.x  
- **Libraries**:  
  - OpenCV  
  - NumPy  
  - OS  
- **Algorithm**: Haarcascade for detection + LBPH for recognition  

---

## 📂 Project Structure  
```
├── dataset/              # Training images for each person
│   ├── person1/
│   ├── person2/
│   └── ...
├── trainer.yml           # Trained model file
├── haarcascade_frontalface_default.xml  # Haarcascade file
├── train.py              # Script to train dataset
├── detect.py             # Script to detect & recognize faces
├── capture.py            # Script to capture images for dataset
└── README.md             # Project documentation
```

---

## 🚀 Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/priyanshu1258/face-recognition-opencv.git
   cd face-recognition-opencv
   ```

2. Install dependencies:  
   ```bash
   pip install opencv-python numpy
   ```

3. Download the Haar Cascade XML:  
   [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)  
   Place it in the project folder.  

---

## 📖 Usage  

### 1️⃣ Capture dataset images  
```bash
python capture.py
```
- Captures face images from webcam and saves them in `dataset/`.  

### 2️⃣ Train the model  
```bash
python train.py
```
- Trains LBPH recognizer and generates `trainer.yml`.  

### 3️⃣ Run face recognition  
```bash
python detect.py
```
- Opens webcam and recognizes faces in real-time.  

---

## 📊 Example Output  
- **Face Detection:** Draws a rectangle around faces.  
- **Face Recognition:** Displays the name/ID of the person on detection.  

---

## 👨‍💻 Team Work Distribution  
- **Member 1:** Dataset collection (images capture).  
- **Member 2:** Preprocessing (grayscale conversion, face detection).  
- **Member 3:** Training model with LBPH.  
- **Member 4:** Real-time detection & recognition script.  
- **Member 5:** Testing and accuracy evaluation.  
- **Member 6:** Documentation (README, project report, presentation).  

---

## 🔮 Future Improvements  
- Improve accuracy using **Deep Learning (OpenCV DNN / FaceNet)**.  
- Add GUI for user-friendly interface.  
- Store data in a database instead of local folders.  

---

## 🤝 Contributing  
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.  

---

## 📜 License  
This project is licensed under the MIT License.  
