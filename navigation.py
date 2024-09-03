import cv2
import numpy as np

## Modify the np array print output to full width instead of it's default format
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))


def find_error(observation, previous_error):
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

    cropped = observation[63:65, 24:73]

    future_road = observation[83:84, 30:66]
    future_road_processing = canny_edge_detector(blur_image(gray_scale(green_mask(future_road))))
    print(future_road_processing)

    # green = green_mask(cropped)
    # grey  = gray_scale(green)
    # blur  = blur_image(grey)
    # canny = canny_edge_detector(blur)

    image_processing = canny_edge_detector(blur_image(gray_scale(green_mask(cropped))))

    # find all non zero values in the cropped strip.
    # These non zero points(white pixels) corresponds to the edges of the road
    nz = cv2.findNonZero(image_processing)

    # horizontal cordinates of center of the road in the cropped slice
    mid = 24

    # some further adjustments obtained through trial and error
    if nz[:, 0, 0].max() == nz[:, 0, 0].min():
        if nz[:, 0, 0].max() < 30 and nz[:, 0, 0].max() > 20:
            return previous_error
        if nz[:, 0, 0].max() >= mid:
            return (-15)
        else:
            return (+15)
    else:
        return (((nz[:, 0, 0].max() + nz[:, 0, 0].min()) / 2) - mid)


def pid(error, previous_error):
    Kp = 0.05
    Ki = 0.025
    Kd = 0.1

    return Kp * error + Ki * (error + previous_error) + Kd * (error - previous_error)
