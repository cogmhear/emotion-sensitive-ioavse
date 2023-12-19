
import numpy as np
import dlib
import cv2
import argparse
import os
from image_utility import save_image, generate_random_color, draw_border
from imutils import face_utils
import skvideo
skvideo.setFFmpegPath('C:/Users/sypdb/.conda/envs/dlib_env/Library/bin/')
print(skvideo.getFFmpegPath())
import skvideo.io
import pdb

def hog_landmarks(image, gray):
    faces_hog = face_detector(gray, 1)

    # HOG + SVN
    for (i, face) in enumerate(faces_hog):
        # Finding points for rectangle to draw on face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Drawing simple rectangle around found faces
        cv2.rectangle(image, (x, y), (x + w, y + h), generate_random_color(), 2)

        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def cnn_landmarks(image, gray):
    faces_cnn = face_detector(gray, 1)

    # CNN
    for (i, face) in enumerate(faces_cnn):
        # Finding points for rectangle to draw on face
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()

        # Drawing simple rectangle around found faces
        cv2.rectangle(image, (x, y), (x + w, y + h), generate_random_color(), 2)

        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, face.rect)
        shape = face_utils.shape_to_np(shape)
        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def dl_landmarks(image, gray, h, w):
    # # This is based on SSD deep learning pretrained model

    # https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
    inputBlob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1, (300, 300), (104, 177, 123))

    face_detector.setInput(inputBlob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]):

        # Probability of prediction
        prediction_score = detections[0, 0, i, 2]
        if prediction_score < args.thresold:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # For better landmark detection
        y1, x2 = int(y1 * 1.15), int(x2 * 1.05)

        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
        shape = face_utils.shape_to_np(shape)
        cv2.rectangle(image, (x1, y1), (x2, y2), generate_random_color(), 2)
        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def face_detection(image):

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time. This will make everything bigger and allow us to detect more
    # faces.

    # write at the top left corner of the image
    img_height, img_width = image.shape[:2]
    if model == 'hog':
        hog_landmarks(image, gray)
    elif model == 'cnn':
        cnn_landmarks(image, gray)
    else:
        dl_landmarks(image, gray, img_height, img_width)

    cv2.putText(image, "68 Pts - {}".format(model), (img_width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                generate_random_color(), 2)

    return image
    '''
    pdb.set_trace()
    save_image(image)

    # Show the image
    cv2.imshow("Facial Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
if __name__ == "__main__":

    #HOME = "/home/keyur-r/image_data"

    # handle command line arguments
    ap = argparse.ArgumentParser()
    #ap.add_argument('-i', '--image', required=True, help='Path to image file')
    ap.add_argument("-l", "--learning", default="hog",
                    help="Which learning model from hog/dl/cnn to use for FaceDetection!")

    ap.add_argument('-w', '--weights',
                    default='./shape_predictor_68_face_landmarks.dat', help='Facial Landmarks Model')
    ap.add_argument('-d', '--data', help='CNN trained model',
                    default='./mmod_human_face_detector.dat')
    ap.add_argument("-p", "--prototxt", default="./deploy.prototxt.txt",
                    help="Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="./res10_300x300_ssd_iter_140000.caffemodel",
                    help="Pre-trained caffe model")
    ap.add_argument("-t", "--thresold", type=float, default=0.6,
                    help="Thresold value to filter weak detections")
    args = ap.parse_args()

    # whether it's hog or cnn or dl
    model = args.learning.lower()

    if model == 'hog':
        # initialize hog + svm based face detector
        face_detector = dlib.get_frontal_face_detector()
    elif model == 'cnn':
        # initialize cnn based face detector with the weights
        face_detector = dlib.cnn_face_detection_model_v1(args.data)
    elif model == 'dl':
        # Pre-trained caffe deep learning face detection model (SSD)
        face_detector = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)
    else:
        print("Please provide valid model name like cnn or hog")
        exit()

    # landmark predictor
    `[3q[]] = dlib.shape_predictor(args.weights)

    filepath= 'C:/Experiments/Datasets/AV_dataset/AV_challenge_dataset/Enhanced_data/ICASSP2023/UNet/A-only/devset/full_face/CC_Loss/EP10/S02082_silent.mp4'
    videogen = skvideo.io.vread(filepath)
    # print(videogen)
    # print"videogen type",videogen.dtype
    frames = np.array([frame for frame in videogen])
    win = dlib.image_window()
    i = 0
    for frame in frames:
            #image = cv2.imread(frame)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_detect = face_detection(frame)
            
            img = face_detect
            # to clear the previous overlay. Useful when multiple faces in the same photo
            win.clear_overlay()

            # to show the image
            win.set_image(img)

            # Ask the detector to find the bounding boxes of each face. The 1 in the
            # second argument indicates that we should upsample the image 1 time. This
            # will make everything bigger and allow us to detect more faces.
            dets = detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, d.left(), d.top(), d.right(), d.bottom()))
                # Get the landmarks/parts for the face in box d.
                shape = predictor(img, d)
                i += 1
                # The next lines of code just get the coordinates for the mouth
                # and crop the mouth from the image.This part can probably be optimised
                # by taking only the outer most points.
                xmouthpoints = [shape.part(x).x for x in range(48,67)]
                ymouthpoints = [shape.part(x).y for x in range(48,67)]
                maxx = max(xmouthpoints)
                minx = min(xmouthpoints)
                maxy = max(ymouthpoints)
                miny = min(ymouthpoints) 

                # to show the mouth properly pad both sides
                pad = 10
                # basename gets the name of the file with it's extension
                # splitext splits the extension and the filename
                # This does not consider the condition when there are multiple faces in each image.
                # if there are then it just overwrites each image and show only the last image.
                filename = os.path.splitext(os.path.basename(f))[0]

                crop_image = img[miny-pad:maxy+pad,minx-pad:maxx+pad]
                cv2.imshow('mouth',crop_image)
                # The mouth images are saved in the format 'mouth1.jpg, mouth2.jpg,..
                # Change the folder if you want to. They are stored in the current directory
                cv2.imwrite(filename+'.jpg',crop_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                win.add_overlay(shape)

            win.add_overlay(dets)



    # if image is valid or not
    image = None
    if args.image:
        # load input image
        img = os.path.join(HOME, args.image)
        image = cv2.imread(img)

    if image is None:
        print("Please provide image ...")
    else:
        print("Face detection for image")
        face_detection(image)


'''
if __name__ == '__main__':

    python facial_landmarks.py -l hog -i <image-path>

    FACE_PREDICTOR_PATH = 'C:/Experiments/Codes/COGMhear_AV_Challenge/AVSE/DeepCCA_AVSE/AVSE_AttnUNet/shape_predictor_68_face_landmarks.dat'
    filepath= 'C:/Experiments/Datasets/AV_dataset/AV_challenge_dataset/Enhanced_data/ICASSP2023/UNet/A-only/devset/full_face/CC_Loss/EP10/S02082_silent.mp4'
    # process video to frames
    print("process video to frames")
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH).from_video(filepath)
    print("jjjj: {}".format(filepath_wo_ext))
'''