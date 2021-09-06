import dlib

# Window
img_filename = 'E:\\LURIE RESEARCH 08 04 21\\FINAL COHORT 8 17 21\\CCHS\\Caucasian\\2-3\\2087.jpg'
img = dlib.load_rgb_image(img_filename)
win = dlib.image_window(img, 'Landmarked Image')
# Detection
detector = dlib.get_frontal_face_detector()
faces = detector(img)
win.add_overlay(faces)
# Prediction
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
for face in faces:
    landmarks = predictor(img, face)
    for part in landmarks.parts():
        win.add_overlay_circle(part, 2)
        #time.sleep(0.2)
win.wait_until_closed()
