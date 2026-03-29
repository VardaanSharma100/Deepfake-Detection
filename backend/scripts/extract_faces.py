import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

input_real = "data/image/raw/real"
input_fake = "data/image/raw/fake"

output_real = "data/image/processed/real"
output_fake = "data/image/processed/fake"



detector = MTCNN()

def extract_and_save_faces(input_folder, output_folder):
    for filename in tqdm(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            results = detector.detect_faces(img)

            for i, face in enumerate(results):
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)

                face_img = img[y:y+h, x:x+w]

                if face_img.size == 0:
                    continue

                save_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{i}.jpg")
                cv2.imwrite(save_path, face_img)

        except Exception as e:
            print(f"Error with {img_path}: {e}")

extract_and_save_faces(input_real, output_real)
extract_and_save_faces(input_fake, output_fake)

print("✅ Face extraction complete! Cropped faces saved in 'processed/'")
