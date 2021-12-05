#Focal Length (EFL): 3.34mm

import cv2
import threading
import numpy as np
import time
import os
import json

from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration

from datetime import datetime

left_camera = None
right_camera = None

cam_width = 640
cam_height = 480

dm_colors_autotune = False
disp_max = -100000
disp_min = 10000

class USB_Camera:

    def __init__ (self) :

        self.video_capture = None
        self.frame = None
        self.grabbed = False
        
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False


    def open(self, cam_id):
        try:
            self.video_capture = cv2.VideoCapture(cam_id)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
            
            
        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: {}".format(cam_id))
            return
        # Grab the first frame to start the video capturing
        self.grabbed, self.frame = self.video_capture.read()


    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running=True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running=False
        self.read_thread.join()

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed=grabbed
                    self.frame=frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened
        

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            frame = cv2.resize(frame, (320,240))
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grabbed=self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


def stereo_depth_map(rectified_pair):
    global disp_max, disp_min, sbm
    
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]

    disparity = sbm.compute(dmLeft, dmRight)

    local_max = disparity.max()
    local_min = disparity.min()
    
    if (dm_colors_autotune):
        disp_max = max(local_max,disp_max)
        disp_min = min(local_min,disp_min)
        local_max = disp_max
        local_min = disp_min
        print(disp_max, disp_min)
        
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET) #HOT, OCEAN, BONE
        
    return disparity_color


def load_map_settings(fName):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, sbm
    print('Loading parameters from file...')
    f=open(fName, 'r')
    data = json.load(f)
    SWS=data['SADWindowSize']
    PFS=data['preFilterSize']
    PFC=data['preFilterCap']
    MDS=data['minDisparity']
    NOD=data['numberOfDisparities']
    TTH=data['textureThreshold']
    UR=data['uniquenessRatio']
    SR=data['speckleRange']
    SPWS=data['speckleWindowSize']
    print('SWS={} PFS={} PFC={} MDS={} NOD={} TTH={} UR={} SR={} SPWS={}'.format(SWS,PFS,PFC,MDS,NOD,TTH,UR,SR,SPWS))
    #sbm.setSADWindowSize(SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print ('Parameters loaded from file '+fName)


def start_cameras(calibration):
    left_camera = USB_Camera()
    left_camera.open(0)
    left_camera.start()

    right_camera = USB_Camera()
    right_camera.open(1)
    right_camera.start()

    cv2.namedWindow("USB Cameras", cv2.WINDOW_AUTOSIZE)

    if ( not left_camera.video_capture.isOpened() or not right_camera.video_capture.isOpened() ):
        # Cameras did not open, or no camera attached
        print("Unable to open any cameras")
        # TODO: Proper Cleanup
        SystemExit(0)

    #disparity = np.zeros((cam_width, cam_height), np.uint8)
    #sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)

    pTime = 0
    imgCount = 0
    while cv2.getWindowProperty("USB Cameras", 0) >= 0 :

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        _ , left_image=left_camera.read()
        _ , right_image=right_camera.read()
        
        left_image= cv2.flip(left_image,-1) # cv2.rotate(right_image, cv2.ROTATE_180)

        left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        if mode_sobel:
                sobelX = cv2.Sobel(left_image_gray, cv2.CV_16S, 1, 0, ksize=3)
                sobelY = cv2.Sobel(left_image_gray, cv2.CV_16S, 0, 1, ksize=3)
                sobelX = np.uint8(np.absolute(sobelX))
                sobelY = np.uint8(np.absolute(sobelY))
                left_image_gray = cv2.bitwise_or(sobelX, sobelY)

                sobelX = cv2.Sobel(right_image_gray, cv2.CV_16S, 1, 0, ksize=3)
                sobelY = cv2.Sobel(right_image_gray, cv2.CV_16S, 0, 1, ksize=3)
                sobelX = np.uint8(np.absolute(sobelX))
                sobelY = np.uint8(np.absolute(sobelY))
                right_image_gray = cv2.bitwise_or(sobelX, sobelY)
        

        rectified_pair = calibration.rectify((left_image_gray, right_image_gray))

        disparity = stereo_depth_map(rectified_pair)
        
        camera_images = np.hstack((left_image, right_image, disparity))
        cv2.putText(camera_images, f'FPS: {int(fps)}', (10, 20), cv2.FONT_ITALIC, 0.5, (0,255,0), 1)
        
        cv2.imshow("USB Cameras", camera_images)

        keyCode = cv2.waitKey(30) & 0xFF
        if keyCode == ord('q') or keyCode == ord('Q'):
            break
            

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    mode_sobel = False
    #disparity = np.zeros((cam_width, cam_height), np.uint8)
    sbm = cv2.StereoBM_create(numDisparities=0, blockSize=5)

    load_map_settings('3dmap_set_240p.txt')

    # Import calibration data
    print('[+] Read calibration data and rectifying stereo pair...')
    calibration = StereoCalibration(input_folder='calib_result_240p')
    
    print('[+] Run')
    start_cameras(calibration)
