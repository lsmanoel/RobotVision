import numpy as np
from matplotlib import pyplot as plt
import time
import cv2
from FrameFilter import FrameFilter

class StereoVision():
    def __init__(self):
        pass

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def plot_depth_map(frame_L_input, frame_R_input):
        #===============================================================================
        frame_disparity = StereoVision.stereo_bm(frame_L_input, frame_R_input)

        fig = plt.figure(figsize=(11,22))
        #===============================================================================  
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(frame_L_input, 'gray')
        ax1.grid(True)
        # plt.xticks([])
        # plt.yticks([])
        ax1.set_aspect('auto')
        #===============================================================================  
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(frame_R_input, 'gray')
        ax2.grid(True)
        # plt.xticks([])
        # plt.yticks([])
        ax2.set_aspect('auto')
        #===============================================================================  
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(frame_disparity, 'inferno')
        ax3.grid(True)
        # plt.xticks([])
        # plt.yticks([])
        ax3.set_aspect('auto')
        #===============================================================================
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(frame_disparity, 'gray')
        ax4.grid(True)
        # plt.xticks([])
        # plt.yticks([])
        ax4.set_aspect('auto')
        #===============================================================================         
        fig.subplots_adjust(wspace=0, hspace=0)

        plt.show()

        return frame_disparity


    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def image_depth_map(frame_L_file = "hero_3_L.png", frame_R_file = "hero_3_R.png"):
        frame_L_input = cv2.imread(frame_L_file)
        frame_R_input = cv2.imread(frame_R_file)        

        frame_L_gray = FrameFilter.color_drop(frame_L_input)
        frame_L_blur = FrameFilter.blur(frame_L_gray)

        frame_R_gray = FrameFilter.color_drop(frame_R_input)
        frame_R_blur = FrameFilter.blur(frame_R_gray)

        #=======================================================
        frame_disparity = StereoVision.stereo_bm(frame_L_input, frame_R_input)

        # frame_disparity = StereoVision.plot_depth_map(frame_L_blur, frame_R_blur)


    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def run_depth_map(video_source_L = 4, video_source_R = 2):

        source_L = cv2.VideoCapture(video_source_L)
        source_R = cv2.VideoCapture(video_source_R)

        CAMERA_WIDTH = 640
        CAMERA_HEIGHT = 480

        source_L.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        source_L.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        source_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        source_R.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        source_R.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        source_R.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        #=======================================================
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=25)

        #=======================================================
        mm_1_buffer, mm_1_pt = FrameFilter.init_moving_average(frame_shape = (CAMERA_HEIGHT, CAMERA_WIDTH),
                                                               buffer_size = 3)

        #=======================================================
        while 1:
            if not (source_L.grab() and source_R.grab()):
                print("No more frames")
                break
            ret, frame_L_input = source_L.retrieve()
            ret, frame_R_input = source_R.retrieve()
            
            #ret, frame_L_input = source_L.read()
            #ret, frame_R_input = source_R.read()
     

            frame_L_gray = cv2.cvtColor(frame_L_input, cv2.COLOR_BGR2GRAY)
            frame_L_blur = FrameFilter.blur(frame_L_gray)
            frame_L_crosshairs = FrameFilter.crosshairs(frame_L_blur)

            frame_R_gray = cv2.cvtColor(frame_R_input, cv2.COLOR_BGR2GRAY)
            frame_R_blur = FrameFilter.blur(frame_R_gray) 
            frame_R_crosshairs = FrameFilter.crosshairs(frame_R_blur)

            frame_concat = np.concatenate((frame_L_crosshairs, frame_R_crosshairs), axis=1)

            frame_disparity = stereo.compute(frame_L_blur, frame_R_blur)
            
            frame_mm_disparity, mm_1_buffer, mm_1_pt = FrameFilter.moving_average(frame_disparity, mm_1_buffer, mm_1_pt)
 
            cv2.imshow("Video", frame_concat)
            # frame_mm_disparity = cv2.applyColorMap(np.uint8(frame_mm_disparity), cv2.COLORMAP_HOT)
            cv2.imshow("disparity", frame_mm_disparity/256)
            
            # --------------------------------------------------
            # Esc -> EXIT while
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                print(frame_disparity.shape)
                break
            # --------------------------------------------------



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#Framefilter.testbench(video_source="./dataset/drone_1.mp4")
#Framefilter.testbench(video_source="./dataset/driver_3.mp4")
# Framefilter.testbench(video_source=2)
StereoVision.run_depth_map()
#StereoVision.image_depth_map(frame_L_file = "ambush_5_left.jpg", frame_R_file = "ambush_5_right.jpg")
#StereoVision.image_depth_map(frame_L_file = "hero_3_L.png", frame_R_file = "hero_3_R.png")