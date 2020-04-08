import cv2

import numpy as np

from centroid import find_centroid
from structural_similarity import compare_ssim as ssim
from functions import inverse_threshold
from homography import transform_image, calculate_homography
from morphological_operations import opening, convert_mask

# Global Variables
roi_h = 300
roi_w = 300
pts_src = np.array([[629, 361], [1066, 411], [396, 486], [946, 571]])
pts_dst = np.array([[0, 0], [roi_w - 1, 0], [0, roi_h - 1], [roi_w - 1, roi_h - 1]])
roi = np.zeros((roi_h, roi_w, 3), np.uint8)
cars = []
next_id = 0

# Reading a video
input_video = cv2.VideoCapture('D:/pytes/Real-Time-Speed-Estimation-master/REPUBLIK RAWIT JENDRAL CABE - SAPEK BANYUWANGI.mp4')

# Calculating homography
hom = calculate_homography(pts_dst, pts_src)

# Reading the empty road frame
ret, road_empty = input_video.read()

roi_empty = np.zeros_like(roi)
transform_image(road_empty, roi_empty, hom)
roi_empty_gray = cv2.cvtColor(roi_empty, cv2.COLOR_BGR2GRAY)
roi_empty_gray = cv2.medianBlur(roi_empty_gray, 11)

# Working with each video frame
while input_video.isOpened():
    next_cars = []

    # Reading a frame from video
    ret, frame = input_video.read()
    if not ret:
        break

    transform_image(frame, roi, hom)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.medianBlur(roi_gray, 11)

    # Calculating the difference between images
    score, diff = ssim(roi_empty_gray, roi_gray)
    roi_diff = diff * 255
    np.clip(roi_diff, 0, 255, out=roi_diff)
    roi_diff = roi_diff.astype('uint8')

    # Threshold an image
    roi_thresh = inverse_threshold(roi_diff, 164)

    # Applying morphological operations
    # roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=3)
    roi_thresh = opening(roi_thresh, convert_mask(np.ones((7, 7), np.uint8)), iterations=3)

    # Getting the contours of an image
    cnts, hier = cv2.findContours(roi_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detecting cars
    for contour in cnts:
        (x, y, w, h) = cv2.boundingRect(contour)
        # Check if the size of contour is big enough
        if w * h > 1200:
            # Calculating centroids
            cx, cy = find_centroid(contour)

            # Defining a car
            new_car = {
                'id': next_id,
                'cx': cx,
                'cy': cy,
                'topLeft': (x, y),
                'bottomRight': (x + w, y + h),
                'speed': -1
            }

            # Check if car was already defined
            for car in cars:
                if abs(car['cx'] - cx) <= 10 and abs(car['cy'] - cy) <= 30:
                    new_car['id'] = car['id']

                    # Calculating the speed of a car (1 px/frame = 9 km/h)
                    speed = abs(car['cy'] - cy) * 9

                    if speed > 0 > car['speed']:
                        new_car['speed'] = speed
                    elif car['speed'] >= 0:
                        new_car['speed'] = (speed + car['speed']) / 2

            # Append a car for the next iteration
            next_cars.append(new_car)

            # Increment Next Car ID if this car is new
            if new_car['id'] == next_id:
                next_id += 1

    cars = next_cars

    # Drawing a rectangle around a car and displaying info (id, speed)
    for car in cars:
        cv2.rectangle(roi, car['topLeft'], car['bottomRight'], (0, 255, 0), 1)

        if car['speed'] is not None:
            cv2.putText(roi, 'id: {}'.format(car['id']), (car['topLeft'][0], car['topLeft'][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.putText(roi, '{0:.0f} km/h'.format(car['speed']), (car['topLeft'][0], car['bottomRight'][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Displaying the playback
    cv2.imshow('Input Video', frame)
    cv2.imshow('ROI', roi)
    cv2.imshow('ROI Empty Road', roi_empty_gray)
    cv2.imshow('ROI Gray', roi_gray)
    cv2.imshow('ROI Difference', roi_diff)
    cv2.imshow('ROI Threshold', roi_thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing and closing resources
input_video.release()
cv2.destroyAllWindows()
