"""
Attempt at making the box2d car follow a slightly more optimal racing line
A controller will try to keep the car on the line
"""

import gymnasium as gym
from visual_processing import green_mask, gray_scale, blur_image, canny_edge_detector, pid  # image processing
from gym.wrappers.monitoring.video_recorder import VideoRecorder  # Video recording
import cv2
import numpy as np


## Modify the np array print output to full width instead of it's default format
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))


def find_error(observation, previous_error):
    '''
    find the error between the cars location and the midline of the road
    :param observation: 96x96 RGB input
    :param previous_error: distance from the road midline (integer)
    :return:
    '''
    cropped = observation[53:55, 24:73]  # for error detection from midline

    image_processing = canny_edge_detector(blur_image(gray_scale(green_mask(cropped))))

    # find all non-zero values in the cropped strip. The non zero points(white pixels) are the edges of the road
    nz = cv2.findNonZero(image_processing)

    # horizontal coordinates of center of the road in the cropped slice. the car is always centered at x=24
    mid = 24

    if nz[:, 0, 0].max() == nz[:, 0, 0].min():  # compare the min and max x coordinates of the road lines
        # if the car is off road
        if 30 > nz[:, 0, 0].max() > 20:
            return previous_error
        if nz[:, 0, 0].max() >= mid:
            return -15
        else:
            return +15
    else:
        # returns the difference between the midline according to the road (average value) and the cars location
        return ((nz[:, 0, 0].max() + nz[:, 0, 0].min()) / 2) - mid


def detect_turn(observation):
    '''
    Detect if one of the road lines has disappeared (the road has turned)
    :param observation: 96x96 RGB input
    :return: Number of turns: integer
    '''
    # future_road = copy.deepcopy(observation[93:94, 24:73])
    future_road = observation[83:84, 30:66]
    future_road_processing = canny_edge_detector(blur_image(gray_scale(green_mask(future_road))))
    print(future_road_processing)
    # print(f"Num Turns {np.count_nonzero(future_road_processing[0])}")
    return np.count_nonzero(future_road_processing[0])


def detect_sharp_turns(observation):
    '''
    Detect if there are sharp turns by comparing early midline shifts to future midline shifts
    :param observation: 96x96 RGB input
    :return: the distance of the upcoming sharp turn (float)
    '''
    # first_detection = observation[53:55, 0:96]
    # second_detection = observation[63:65, 0:96]  # for further error detection from midline

    # first_detection = observation[73:75, 0:96]
    # second_detection = observation[81:83, 0:96]  # for further error detection from midline

    first_detection = observation[44:46, 0:96]
    second_detection = observation[53:55, 0:96]  # for further error detection from midline

    first_processing = canny_edge_detector(blur_image(gray_scale(green_mask(first_detection))))
    second_processing = canny_edge_detector(blur_image(gray_scale(green_mask(second_detection))))

    fnz = cv2.findNonZero(first_processing)
    snz = cv2.findNonZero(second_processing)
    print(f"sharp_turn: {((fnz[:, 0, 0].max() + fnz[:, 0, 0].min()) / 2) - ((snz[:, 0, 0].max() + snz[:, 0, 0].min()) / 2)}")
    # print(f"midline of first detection: {(fnz[:, 0, 0].max() + fnz[:, 0, 0].min()) / 2}")
    # print(f"midline of second detection: {(snz[:, 0, 0].max() + snz[:, 0, 0].min()) / 2}")

    # print(np.count_nonzero(second_processing[0]))

    if np.count_nonzero(second_processing[0]) < 2 and np.count_nonzero(first_processing[0]) == 1:  # if the roadline disappears in the second detection, please turn
        return True

    if abs((fnz[:, 0, 0].max() + fnz[:, 0, 0].min()) / 2) - ((snz[:, 0, 0].max() + snz[:, 0, 0].min()) / 2) > 5:
        return True


def speed_detection(observation):
    pass


env = gym.make("CarRacing-v2", render_mode="rgb_array", lap_complete_percent=1)

# Save the video to the directory as test.mp4
video = VideoRecorder(env, "test_final.mp4", enabled=True)
observation = env.reset()
# turn_obs = copy.deepcopy(observation)
env.render()
rewards = 0
previous_error = 0
turning = False
sharp_turn = False

turn1_count, turn2_count, turn3_count = 0, 0, 0
sharpest_turn = 0
escape_turn = 0

for x in [1, 1, 1, 0] * 270:
    video.capture_frame()
    try:
        error = find_error(observation, previous_error)
        num_turns = detect_turn(observation)
        sharp_turn = detect_sharp_turns(observation)
    except Exception as error_message:  # I think this handles the very first frame where a type error occurs
        error = -15
        num_turns = 2
        print(error_message)
        pass

    steering = pid(error, previous_error)
    action = (steering, x, 0)  # steering input, throttle, brake

    if turning or sharp_turn:
        if sharp_turn and num_turns > 2:
            sharpest_turn += 1
            action = (steering, 0.1, 0.6)
            sharp_turn = False
            turning = False

        elif sharp_turn:
            turn2_count += 1
            action = (steering, 0.2, 0.3)
            sharp_turn = False
            turning = False

    elif abs(action[0]) < 0.01 or not sharp_turn:  # ignore minor steering corrections, drive straight
        action = (0, x, 0)
        if abs(action[0]) > 0.8 or sharp_turn or num_turns == 1 or num_turns >= 3:
            turning = True
    print(f"Steering: {action[0]}")
    new_action = env.step(action)[:-1]
    observation, reward, done, info = new_action
    previous_error = error
    rewardsum = rewards + reward

    if done:
        video.close()
        env.close()
        break

print("reward", rewardsum)
video.close()
env.close()
print(f"Roadline disappeared: {turn1_count}, Sharp turn: {turn2_count}, Post Turn: {turn3_count}")
print(f"Sharpest turns: {sharpest_turn}")
