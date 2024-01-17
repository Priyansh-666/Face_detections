import cv2
import os
import face_recognition
import time
import pickle
import dlib
import numpy as np

face_detector = dlib.get_frontal_face_detector()
def enhance_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    min_intensity = gray.min()
    max_intensity = gray.max()
    stretched = ((gray - min_intensity) / (max_intensity - min_intensity) * 255).astype(np.uint8)

    equalized = cv2.equalizeHist(stretched)

    return equalized

with open('encodings.pkl', 'rb') as f:
    encodeListKnown = pickle.load(f)

known_faces_path = './image'  
test_images_path = './group' 
classNames = [os.path.splitext(cl)[0] for cl in os.listdir(known_faces_path)]
print(classNames)

for filename in os.listdir(test_images_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(test_images_path, filename))

        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print(f"Error: Empty frame received from {filename}.")
            continue

        face_locations = face_detector(img,1)

        confidence_threshold = 0.55 

        for face in face_locations:
            l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
            face = img[t:b, l:r] 

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face_encodings = face_recognition.face_encodings(face_rgb)

            if len(face_encodings) > 0:
                encodeFace = face_encodings[0]  
                
                t1 = time.time()
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if faceDis[matchIndex] < confidence_threshold:
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        confidence = 1 - faceDis[matchIndex]
                        t2 = time.time()
                        cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
                        cv2.rectangle(img, (l, b - 35), (r, b), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, f"{name}", (l + 6, b - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        if not os.path.exists(f"./save_group/group/{filename.split('.')[0]}/detected"):
                            os.makedirs(f"./save_group/group/{filename.split('.')[0]}/detected")
                        cv2.imwrite(f"./save_group/group/{filename.split('.')[0]}/detected/{name}.jpg", face_rgb)
                else:
                    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.rectangle(img, (l, b - 35), (r, b), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, f"unknown", (l + 6, b - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    if not os.path.exists(f"./save_group/group/{filename.split('.')[0]}/not_detected"):
                        os.makedirs(f"./save_group/group/{filename.split('.')[0]}/not_detected")
                    cv2.imwrite(f"./save_group/group/{filename.split('.')[0]}/not_detected/unknown.jpg",face_rgb)
          
        cv2.imwrite(f"./save_group/{filename}",img)

cv2.destroyAllWindows()
