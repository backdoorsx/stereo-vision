#Focal Length (EFL): 3.34mm

import cv2
import threading
import numpy as np
import time
import os

left_camera = None
right_camera = None

cam_width = 640
cam_height = 480

img_width = 320
img_height = 240

img_size = (img_width, img_height)

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


def start_cameras():
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

    pTime = 0
    imgCount = 0
    while cv2.getWindowProperty("USB Cameras", 0) >= 0 :
        
        _ , left_image=left_camera.read()
        _ , right_image=right_camera.read()

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        left_image = cv2.resize(left_image, (img_width, img_height))
        right_image = cv2.resize(right_image, (img_width, img_height))

        left_image = cv2.flip(left_image,-1) # cv2.rotate(right_image, cv2.ROTATE_180)
        
        if mode_sobel:
                sobelX = cv2.Sobel(left_image, cv2.CV_16S, 1, 0, ksize=3)
                sobelY = cv2.Sobel(left_image, cv2.CV_16S, 0, 1, ksize=3)
                sobelX = np.uint8(np.absolute(sobelX))
                sobelY = np.uint8(np.absolute(sobelY))
                left_image = cv2.bitwise_or(sobelX, sobelY)

                sobelX = cv2.Sobel(right_image, cv2.CV_16S, 1, 0, ksize=3)
                sobelY = cv2.Sobel(right_image, cv2.CV_16S, 0, 1, ksize=3)
                sobelX = np.uint8(np.absolute(sobelX))
                sobelY = np.uint8(np.absolute(sobelY))
                right_image = cv2.bitwise_or(sobelX, sobelY)
        
        camera_images = np.hstack((left_image, right_image))
        cv2.putText(camera_images, f'FPS: {int(fps)}', (10, 20), cv2.FONT_ITALIC, 0.5, (0,255,0), 1)

        cv2.imshow("USB Cameras", camera_images)

        # This also acts as
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == ord('q'):
            break
        if keyCode == ord('s'):
            if os.path.isdir('./depth_tune_240p') == False:
                os.mkdir('./depth_tune_240p')
            
            print('Save imgs : {}'.format(imgCount))
            filename_left = './depth_tune_240p/left_'+str(imgCount)+'.png'
            filename_right = './depth_tune_240p/right_'+str(imgCount)+'.png'
            print('  Left img: {}'.format(filename_left))
            print('  Right img: {}'.format(filename_right))
            cv2.imwrite(filename_left, left_image)
            cv2.imwrite(filename_right, right_image)
            imgCount = imgCount +1
            

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode_sobel = False
    start_cameras()
