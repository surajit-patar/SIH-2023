# Import TF and TF Hub libraries.
import tensorflow as tf
import numpy as np

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display
import os
import cv2
# for k, v in os.environ.items():
#     if k.startswith("QT_") and "cv2" in v:
#         del os.environ[k]


def get_pose(image_path):
    # Dictionary that maps from joint names to keypoint indices.
    KEYPOINT_DICT = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }

    # Maps bones to a matplotlib color name.
    KEYPOINT_EDGE_INDS_TO_COLOR = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
    }



    # Load the input image.
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    # image = tf.image.resize_with_pad(image, 192, 192) ## LIGHTNING
    image = tf.image.resize_with_pad(image, 256, 256) ## THUNDER

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="./MODELS/single_pose_thunder.tflite")
    interpreter.allocate_tensors()

    # TF Lite format expects tensor type of float32.
    input_image = tf.cast(image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

    interpreter.invoke()

    # Output is a [1, 1, 17, 3] numpy array.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])[0][0]

    print(keypoints_with_scores)


    orig = cv2.imread(image_path)
    imH, imW,_ = orig.shape

    key_list = [(k, v) for k, v in KEYPOINT_DICT.items()]
    confidense_threhold = 0.55

    for i in range(17):
        ymin = int(max(1,(keypoints_with_scores[i][0] * imH)))
        xmin = int(max(1,(keypoints_with_scores[i][1] * imW)))
        confidense = keypoints_with_scores[i][2]
        if confidense>confidense_threhold:
            orig = cv2.circle(orig, (xmin,ymin), 5, (255,0,0), -1)
            print(key_list[i])

    ## draw skeleton
    for i in range(17):
        ymin = int(max(1,(keypoints_with_scores[i][0] * imH)))
        xmin = int(max(1,(keypoints_with_scores[i][1] * imW)))
        confidense = keypoints_with_scores[i][2]
        if confidense>confidense_threhold:
            orig = cv2.circle(orig, (xmin,ymin), 5, (255,0,0), -1)
            print(key_list[i])
            for j in range(17):
                if i!=j and confidense>confidense_threhold:
                    ymin2 = int(max(1,(keypoints_with_scores[j][0] * imH)))
                    xmin2 = int(max(1,(keypoints_with_scores[j][1] * imW)))
                    confidense2 = keypoints_with_scores[j][2]
                    if confidense2>confidense_threhold:
                        orig = cv2.line(orig, (xmin,ymin), (xmin2,ymin2), (0,255,0), 2)
                        print(key_list[i], key_list[j])

                    
    cv2.imwrite("output.jpg", orig)
    

