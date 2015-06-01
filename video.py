import cv2

path = '/home/rshir/data/simple_sift/result.avi'  # Video path
number_of_init_frames = 10  # Number of frames for initialization stage


# Open the video and read first fame
def open_video(path):
    cap = cv2.VideoCapture(path)
    # take first frame of the video
    if cap.isOpened(): # try to get the first frame
        ret, frame = cap.read()
    else:
        ret = False
    return ret, cap


def get_frame(ret, cap):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return ret, gray
    else:
        return ret, None


def get_frame_1(ret, cap):
    return get_frame(ret, cap)


def get_frame_n(ret, cap, n):
    cap = cv2.VideoCapture(path)
    for i in range(1, n):
        ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return ret, gray


