import cv2
import numpy as np


def green_mask(observation):
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))

    ## slice the green
    imask_green = mask_green > 0
    green = np.zeros_like(observation, np.uint8)
    green[imask_green] = observation[imask_green]
    return (green)


def gray_scale(observation):
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    return gray


def blur_image(observation):
    blur = cv2.GaussianBlur(observation, (5, 5), 0)
    return blur


def canny_edge_detector(observation):
    canny = cv2.Canny(observation, 50, 150)
    return canny

def pid(error,previous_error):
    # Kp = 0.035
    # Ki = 0.015
    # Kd = 0.15

    Kp = 0.055
    Ki = 0.02
    Kd = 0.15


    return Kp * error + Ki * (error + previous_error) + Kd * (error - previous_error)
