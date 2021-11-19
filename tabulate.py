# Native
import csv
import itertools
import os
from pathlib import Path
# External
import dlib
import cv2

# Description: Landmarks frontal photos & writes to a CSV file.

# Configuration
RESULT_FILE = 'data/samples_updated.csv'
SHAPE_PREDICTOR_MODEL = 'models/shape_predictor_68_face_landmarks.dat'

# Globals
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_MODEL)

with open(RESULT_FILE, 'w', newline='') as result_file:
    result_writer = csv.writer(result_file)
    # Header row
    header_row = ['filename', 'subject', 'group', 'race', 'age']
    for i in range(68):
        header_row.append('x' + str(i))
        header_row.append('y' + str(i))
    result_writer.writerow(header_row)
    # Iterate over image directories
    image_count = 0
    for group in ('CONTROL', 'CCHS'):
        for race in ('African American', 'Caucasian', 'Other', 'UNKNOWN'):
            for age in ('0-1', '2-3', '4-5', '6-8', '9-11', '12-14', '15-17', '18-25', '26-35'):
                # Glob image files
                image_path = Path('E:\\Lurie Research\\final cohort3', group, race, age)
                if not os.path.isdir(image_path):
                    continue
                image_filenames = itertools.chain(
                    image_path.glob('*.png'),
                    image_path.glob('*.jpg'),
                    image_path.glob('*.jpeg')
                )
                for image_filename in image_filenames:
                    print(f'{image_count}: {str(image_filename)}', end='\r')
                    # Load image
                    image = dlib.load_rgb_image(str(image_filename))
                    faces = detector(image)
                    # Rotate image into proper orientation
                    rotated_count = 0
                    while rotated_count < 4 and len(faces) == 0:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        faces = detector(image)
                        rotated_count += 1
                    # Landmarking
                    for face in faces:
                        landmarks = predictor(image, face)
                        identifier = str(image_filename).split("_")[-1][:-4]
                        row = [image_filename, identifier, group, race, age]
                        for part in landmarks.parts():
                            row.append(part.x)
                            row.append(part.y)
                        result_writer.writerow(row)
                    image_count += 1
