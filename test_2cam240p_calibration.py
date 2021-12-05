#Focal Length (EFL): 3.34mm

import cv2
import threading
import numpy as np
import time
import os
import json

from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

# Chessboard parameters
rows = 6
columns = 9
square_size = 2.5

# Global variables
left_camera = None
right_camera = None

cam_width = 640
cam_height = 480

img_width = 320
img_height = 240

img_size = (img_width, img_height)
IMGS_TOTAL = 64
AUTO_TIMER = 3.5 # unit second


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
            grabbed=self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


def start_cameras(calibrator):
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

    if os.path.isdir('./calib_img_240p') == False:
        os.mkdir('./calib_img_240p')

    pTime = 0
    pTime_autoTimer = time.time()
    imgCount = 0
    save_tag = False
    while cv2.getWindowProperty("USB Cameras", 0) >= 0 :

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        _ , left_image=left_camera.read()
        _ , right_image=right_camera.read()

        left_image = cv2.resize(left_image, (320,240))
        right_image = cv2.resize(right_image, (320,240))
        
        left_image = cv2.flip(left_image,-1) # cv2.rotate(right_image, cv2.ROTATE_180)

        #print('{} - {} = {}'.format(cTime, pTime_autoTimer, (cTime-pTime_autoTimer)))
        if (cTime - pTime_autoTimer) >= AUTO_TIMER:
            
            try:
                corner_left = calibrator._get_corners(left_image)
                corner_right = calibrator._get_corners(right_image)
                print('[*] calibrator get corners')
            except ChessboardNotFoundError as error:
                #print('[*] chessboard not found')
                pass
            else:
                #show chessboard
                tmp_left = left_image.copy()
                cv2.drawChessboardCorners(tmp_left, (rows, columns), corner_left, True)
                tmp_right = right_image.copy()
                cv2.drawChessboardCorners(tmp_right, (rows, columns), corner_right, True)
            
                #calibrator.add_corners((left_image, right_image), False)
                #print('[*] calibrator add corners')
                save_tag = True
            
            
        if save_tag:
            camera_images = np.hstack((tmp_left, tmp_right))
            txt_save = '[*] press "s" to save or "n" to dont save'
            print(txt_save)
            cv2.putText(camera_images, f'{txt_save}', (10, int(img_height-20)), cv2.FONT_ITALIC, 0.5, (0,0,255), 1)
            
        else:
            camera_images = np.hstack((left_image, right_image))
            
        
        cv2.putText(camera_images, f'FPS: {int(fps)}', (10, 20), cv2.FONT_ITALIC, 0.5, (0,255,0), 1)

        t = (AUTO_TIMER-(cTime-pTime_autoTimer))
        if t > 0:
            cv2.putText(camera_images, f'{int(t)}', (int(img_width-10), int(img_height/2)), cv2.FONT_ITALIC, 1, (0,0,255), 3)

        
        cv2.imshow("USB Cameras", camera_images)


        while save_tag:
            if cv2.waitKey(30) == ord('s'):
                calibrator.add_corners((left_image, right_image), False)
                print('[*] calibrator add corners')
                print('[+] Save imgs : {}'.format(imgCount))
                filename_left = './calib_img_240p/left_'+str(imgCount)+'.png'
                filename_right = './calib_img_240p/right_'+str(imgCount)+'.png'
                cv2.imwrite(filename_left, left_image)
                cv2.imwrite(filename_right, right_image)
                imgCount = imgCount +1
                save_tag = False
                pTime_autoTimer = time.time() # reset auto timer
            elif cv2.waitKey(30) == ord('n'):
                save_tag = False
                pTime_autoTimer = time.time() # reset auto timer
        
        
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == ord('q'):
            break

        if imgCount >= IMGS_TOTAL:
            break


    t = time.time()
    print('[+] Starting calibration... ')
    print('[+] 64 img with in 10 minut about.')
    calibration = calibrator.calibrate_cameras()
    t = time.time() - t
    print('[+] Done! Time of calibration : {}minut {}second'.format(int(t//60),int(t%60)))
    calibration.export('calib_result_240p')
    print('[+] Calibration exported to the folder calib_result')
    print('[+] Calibration complete!')

    # Lets rectify and show last pair after  calibration
    calibration = StereoCalibration(input_folder='calib_result_240p')
    rectified_pair = calibration.rectify((left_image, right_image))

    cv2.imshow('Left CALIBRATED', rectified_pair[0])
    cv2.imshow('Right CALIBRATED', rectified_pair[1])
    cv2.imwrite("rectifyed_left.jpg", rectified_pair[0])
    cv2.imwrite("rectifyed_right.jpg", rectified_pair[1])
    cv2.waitKey(0)
    
    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    calibrator = StereoCalibrator(rows, columns, square_size, img_size)
    print('[+] Run')
    
    start_cameras(calibrator)
