# UCSD ECE MAE 148 - Team 1
#
# Contains the code for defining the Car class. Defines methods
# for throttle and steering depending on DepthAI object and color
# detection. Establishes a data pipeline using OAKD camera.
#
# Sources: 
#   - VESC Class object : https://drive.google.com/drive/folders/1SBzChXK2ebzPHgZBP_AIhVXJOekVc0r3
#   - DepthAI pipeline demo : https://github.com/luxonis/depthai
#   - Traffic light object detection : https://github.com/HevLfreis/TrafficLight-Detector/blob/master/src/main.py

from pathlib import Path
import vesc
import sys
import cv2
import depthai as dai
import numpy as np
import time
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs

# import camera
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# traffic light detection
def detect(frame):

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = frame
    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color range
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    # lower_yellow = np.array([15,100,100])
    # upper_yellow = np.array([35,255,255])
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)

    size = img.shape

    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=10, minRadius=0, maxRadius=30)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=5, minRadius=0, maxRadius=30)

    # traffic light detect
    r = 5
    bound = 4.0 / 10
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))
        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)


    return [r_circles is not None, g_circles is not None, y_circles is not None]

class Car:
    def __init__(self, starting_state, debug, name="Mario") -> None:
        self.states = ["DEFAULT", "FOLLOW", "STOP", "R-UTURN", "L-UTURN", "RTURN", "LTURN"]
        self.name = name
        self.state = starting_state
        self.debug = debug
        self.vesc = vesc.VESC("/dev/ttyACM0")
        self.throttle = 0.0
        self.angle = 0.5
        
    def default(self):
        self.vesc.run(0.5, 0.1)
        self.throttle = 0.1
        self.angle = 0.5
        if (self.debug):
            print("default")
    
    def stop(self):
        self.vesc.run(0.5, 0.0)
        self.throttle = 0.0
        self.angle = 0.5
        if (self.debug):
            print("stopping")
    
    def rturn(self):
        self.vesc.run(0.8, self.throttle)
        self.throttle = self.throttle
        self.angle = 0.8
        if (self.debug):
            print("right turn")
    
    def lturn(self):
        self.vesc.run(0.2, self.throttle)
        self.throttle = self.throttle
        self.angle = 0.2
        if (self.debug):
            print("left turn")
    
    def set_throttle(self, throttle):
        self.vesc.run(self.angle, throttle)
        self.throttle = throttle
        self.angle = self.angle
        if (self.debug):
            print("throttle set to ", throttle)
    
    def up_throttle(self):
        self.set_throttle(self.throttle + 0.05)
        if (self.debug):
            print("upped throttle")
    
    def down_throttle(self):
        self.set_throttle(self.throttle - 0.05)
        if (self.debug):
            print("downed throttle")

    def r_uturn(self):
        self.stop()
        self.vesc.run(0.4, 0.1)
        time.sleep(0.2)
        self.vesc.run(0.9, 0.1)
        time.sleep(0.6)
    
    def l_uturn(self):
            self.stop()
            self.vesc.run(0.6, 0.1)
            time.sleep(0.2)
            self.vesc.run(0.1, 0.1)
            time.sleep(0.6)

    def setDynThrottle(self, zdist):
        if zdist >= 4000:
            self.throttle = 0.3
        elif zdist <= 0:
            self.throttle = 0.0
        else:
            self.throttle = (0.00006 * zdist)

    def setDynSteering(self, mid):
        angle = 0.003*mid
        self.angle = min(0.8, angle)
        self.angle = max(0.1, angle)

                
    def run(self):
        # Get argument first for blob
        nnBlobPath = str((Path(__file__).parent / Path('./mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())

        # MobilenetSSD label texts
        labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        syncNN = True

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutDepth = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")
        xoutDepth.setStreamName("depth")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        if syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        else:
            camRgb.preview.link(xoutRgb.input)

        spatialDetectionNetwork.out.link(xoutNN.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            old_state = [False, True, True]
            state = "STOP"
            while True:
                inPreview = previewQueue.get()
                inDet = detectionNNQueue.get()
                depth = depthQueue.get()

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()
                new_state = detect(frame)

                if new_state[0] == True and new_state[1] != True:
                    state = "STOP"
                elif new_state[1] == True:
                    state = "GO"

                # if the stoplight (red) was detected
                if state == "STOP":
                    self.stop()

                    # if a greenlight was detected
                elif state == "GO":
                    depthFrame = depth.getFrame() # depthFrame values are in millimeters

                    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depthFrameColor = cv2.equalizeHist(depthFrameColor)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                    depthFrameColor = cv2.circle(depthFrameColor, (200,200), 2, (0,0,255), 2)

                    detections = inDet.detections

                    # If the frame is available, draw bounding boxes on it and show the frame
                    height = frame.shape[0]
                    width  = frame.shape[1]
                    for detection in detections:
                        roiData = detection.boundingBoxMapping
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)
                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                        # Denormalize bounding box
                        x1 = int(detection.xmin * width)
                        x2 = int(detection.xmax * width)
                        y1 = int(detection.ymin * height)
                        y2 = int(detection.ymax * height)
                        try:
                            label = labelMap[detection.label]
                        except:
                            label = detection.label
             
                        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                        # If a Person is detected, follow them. Adjust steering and throttle
                        # to get close and stop when too close. 
                        if (label == 'person'):
                            mid = int(((bbox[2]-bbox[0])/2) + bbox[0])
                            self.setDynThrottle(int(detection.spatialCoordinates.z))
                            self.setDynSteering(mid)
                            self.vesc.run(self.angle, self.throttle)

                        # If a person is not detected, then stop.
                        else:
                            self.stop()
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

                    if cv2.waitKey(1) == ord('q'):
                        break
                old_state = new_state