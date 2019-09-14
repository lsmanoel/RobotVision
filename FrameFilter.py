import numpy as np
from matplotlib import pyplot as plt
import time
import cv2

class FrameFilter():
    def __init__(self):
        pass

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Shift the frame along X or Y axis
    @staticmethod
    def translation(frame_input, n_x=5, n_y=0):
        M = np.float32([[1, 0, n_x], [0, 1, n_y]])
        frame_output = cv2.warpAffine(frame_input, M, (frame_input.shape[1], frame_input.shape[0]))
        return frame_output

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Moving Average Filter apply along the time axis
    # Order = buffer_size
    @staticmethod
    def init_moving_average(frame_shape, buffer_size = 50):
        buffer_pt = 0
        frame_buffer = []
        for i in range(buffer_size):
            frame_buffer.append(np.zeros(frame_shape))

        return frame_buffer, buffer_pt

    @staticmethod
    def moving_average(frame_input, frame_buffer, frame_pt):
        frame_output = np.zeros(frame_input.shape)

        if frame_pt > len(frame_buffer) - 1:
            frame_pt = 0

        frame_buffer[frame_pt] = frame_input
        for frame in frame_buffer:
            frame_output = frame_output + frame

        frame_output = frame_output/len(frame_buffer)

        frame_pt = frame_pt + 1

        return frame_output, frame_buffer, frame_pt

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Puts a centered crosshairs on the frame 
    @staticmethod
    def crosshairs(frame_input):
        frame_output = frame_input
        cv2.line(frame_output,
                 (0, frame_output.shape[0]//2),
                 (frame_output.shape[1], frame_output.shape[0]//2),
                 (255,0,0),
                 1)

        cv2.line(frame_output,
                 (frame_output.shape[1]//2, 0),
                 (frame_output.shape[1]//2, frame_output.shape[0]),
                 (255,0,0),
                 1)
        return frame_output

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def horizontal_edges_extraction(frame_input):
        frame_output = cv2.Sobel(frame_input, cv2.CV_32F, 0, 1, -1)
        frame_output = cv2.convertScaleAbs(frame_output)
        frame_output = cv2.GaussianBlur(frame_output, (3, 3), 0)
        ret, frame_output = cv2.threshold(frame_output, 120, 255, cv2.THRESH_BINARY)

        kernel = np.ones((1, 1), np.uint8)
        frame_output = cv2.erode(frame_output, kernel, iterations=1)
        frame_output = cv2.dilate(frame_output, kernel, iterations=2)
        frame_output = cv2.erode(frame_output, kernel, iterations=1)
        frame_output = cv2.dilate(frame_output, kernel, iterations=2)

        return frame_output

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def binarize(frame_input):
        frame_output = cv2.Sobel(frame_input, cv2.CV_32F, 0, 1, -1)
        frame_output = cv2.convertScaleAbs(frame_output)
        frame_output = cv2.GaussianBlur(frame_output, (3, 3), 0)
        ret, frame_output = cv2.threshold(frame_output, 120, 255, cv2.THRESH_BINARY)

        kernel = np.ones((1, 1), np.uint8)
        frame_output = cv2.erode(frame_output, kernel, iterations=1)
        frame_output = cv2.dilate(frame_output, kernel, iterations=2)
        frame_output = cv2.erode(frame_output, kernel, iterations=1)
        frame_output = cv2.dilate(frame_output, kernel, iterations=2)

        return frame_output

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def blur(frame_input, kernel_size=(3, 3)):
        frame_output = cv2.GaussianBlur(frame_input, 
        								kernel_size, 
        								0)
        return frame_output

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def blurSobel(frame_input):
        frame_output = cv2.Sobel(frame_input, cv2.CV_32F, 0, 1, -1)
        frame_output = cv2.convertScaleAbs(frame_output)
        frame_output = FrameFilter.Blur(frame_output)
        return frame_output

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def signature_histogram_generation(frame_input):
        # frame_output = np.zeros((frame_input.shape))
        histogram = np.zeros(frame_input.shape[0])

        for i, line in enumerate(frame_input[:,]):
            line_energy = np.sum(line)**2
            histogram[i] = int(line_energy/(4096**2))

        return histogram

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def plot_line(line_input, frame_output_shape):
        frame_output = np.zeros(frame_output_shape)
        #print(frame_output_shape)
        for i, value in enumerate(line_input):
            value = int(value)
            if value >= frame_output_shape[1]:
                value = frame_output_shape[1]
            #print(int(value))
            frame_output[i,:int(value)] = np.ones(int(value))

        return frame_output

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def display(frame_input, frame_name='frame'):
        frame_output = frame_input
        cv2.imshow(frame_name, frame_input) 

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def print_mean(frame_input):
        value = np.mean(frame_input)
        print(value) 

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def testbench(video_source=0):
        cam = cv2.VideoCapture(video_source)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320.0)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.0)

        #=======================================================
        mm_1_buffer, mm_1_pt = FrameFilter.init_moving_average(frame_shape = (240, 320),
                                                               buffer_size = 50)

        #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        while(1):
            ret, frame_input = cam.read()

            fps = cam.get(cv2.CAP_PROP_FPS)

            #  =============================================
            frame_gray = cv2.cvtColor(frame_input, cv2.COLOR_BGR2GRAY)
            frame_edge = FrameFilter.horizontal_edges_extraction(frame_gray)
            frame_mm, mm_1_buffer, mm_1_pt = FrameFilter.moving_average(frame_edge, mm_1_buffer, mm_1_pt)

            histogram = FrameFilter.signature_histogram_generation(frame_mm)
            frame_histogram = FrameFilter.plot_line(histogram, frame_gray.shape) 
            frame_concat = np.concatenate((frame_edge, frame_histogram), axis=1)
 
            #  =============================================
            Framefilter.display(frame_concat)
            #Framefilter.print_mean(frame_concat)
            
            #  ============================================= 
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        cam.release()
        cv2.destroyAllWindows()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Framefilter.testbench(video_source=2)