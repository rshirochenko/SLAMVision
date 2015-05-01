from numpy import *
from scipy import linalg

class Camera(object):
    """Initialize camera projection matrix"""
    kNear = 0.01
    kFar = 10000

    def __init__(self):
        print("Camera is initiated")
        self.P = None
        self.camera_width = None
        self.camera_height = None
        self.fku = None
        self.fkv = None
        self.u0 = None
        self.v0 = None
        self.kd1 = None
        self.sd = None

    def setCameraParametrs(self):
        print("setting camera parameters")

    def setCameraParameters(self,camera_width,camera_height,fku,fkv,u0,v0,kd1,sd):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fku = fku
        self.fkv = fkv
        self.u0 = u0
        self.v0 = v0
        self.kd1 = kd1
        self.sd = sd

        xW = camera_width * self.kNear/fku
        yH = camera_height * self.kNear/fkv


