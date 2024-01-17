# Face Recognition System

This is a Python script for a simple face recognition system using the `face_recognition` library, OpenCV, and dlib. The script processes a group of images, detects faces, and compares them against a set of known faces, saving the results in the `save_group` directory.

## Prerequisites

Make sure you have the required libraries installed. You can install them using:

```bash
pip install opencv-python face_recognition dlib numpy
```

# How to Run
Clone the repository:

```
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

Place the images of known faces in the image directory.

Place the images to be tested in the group directory.

```
python face_recognition_script.py
```

The script will process the images, detect faces, compare them against the known faces, and save the results in the save_group directory.


## Additional Notes

- The script uses a confidence threshold of 0.55 for face matching. You can adjust this threshold in the script based on your requirements.

- Detected faces are saved in the `detected` subdirectory, while unrecognized faces are saved in the `not_detected` subdirectory within each group's folder.

- Ensure that the necessary directories (`save_group`, `image`, and `group`) exist before running the script.

Feel free to customize the script according to your needs. If you encounter any issues, please check the dependencies and make sure the file paths are correct.

