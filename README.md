# Face Detection with Age & Smile Recognition

This project uses OpenCV and a pretrained Caffe model to:

- Detect faces in real-time using webcam
- Estimate age range of detected faces
- Detect smiles
- Save only unique faces (no duplicates) to disk

## 📂 Files
- sen_unique_faces.py: Main Python script
- age_deploy.prototxt and age_net.caffemodel: Required age estimation model files (not included in this repo)
- faces/: Folder where unique face images are saved

## 🛠 Requirements

Install dependencies:

```bash
pip install opencv-python numpy
```

Download these required model files and place them in the same folder:
 
 • age_deploy.prototxt ( https://github.com/spmallick/learnopencv/blob/master/AgeGender/age_deploy.prototxt )
 
 • age_net.caffemodel ( wget https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/5b60d9f8a8608cdbbcdaaa39bf28f351e8d8553b/model/age_net.caffemodel )


## 🚀 Run

Run python sen_unique_faces.py

Press ENTER to quit the application
## save file📸
All unique detected faces will be saved in the faces/ folder with timestamped filenames.
