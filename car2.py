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

class Car:
    def __init__(self, starting_state, debug, name="Mario") -> None:
        self.states = ["DEFAULT", "FOLLOW", "STOP", "R-UTURN", "L-UTURN", "RTURN", "LTURN"]
        self.name = name
        self.state = starting_state
        self.debug = debug
        self.vesc = vesc.VESC("/dev/ttyACM0")
        self.throttle = 0.0
        self.angle = 0.5
        
        # self.camera or whatever
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
        # if mid >= 130 and mid <= 170:
        #     angle = 0.5
        # else:
        #     angle = 0.003*mid
        angle = 0.003*mid
        self.angle = min(0.8, angle)
        self.angle = max(0.1, angle)

                
    def run(self):
        # # Get argument first
        nnBlobPath = str((Path(__file__).parent / Path('./mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
        # if len(sys.argv) > 1:
        #     nnBlobPath = sys.argv[1]

        # if not Path(nnBlobPath).exists():
        #     import sys
        #     raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

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

                depthFrame = depth.getFrame() # depthFrame values are in millimeters
                # print(np.shape(depthFrame))

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
                    # cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    if (label == 'person'):
                        # print("Dist: ", int(detection.spatialCoordinates.z), " Throttle: ", 0.00022 *int(detection.spatialCoordinates.z))
                        mid = int(((bbox[2]-bbox[0])/2) + bbox[0])
                        self.setDynThrottle(int(detection.spatialCoordinates.z))
                        self.setDynSteering(mid)
                        self.vesc.run(self.angle, self.throttle)
                        # if (mid> 170):
                        #     self.setDynThrottle(int(detection.spatialCoordinates.z))
                        #     print()
                        #     self.angle = 0.5 # 0.7
                        #     self.vesc.run(self.angle, self.throttle)
                        #     print("MID: ", mid, " Steering: ", 0.003*mid)
                        # elif (mid < 130):
                        #     self.setDynThrottle(int(detection.spatialCoordinates.z))
                        #     self.angle = 0.5 # 0.3
                        #     self.vesc.run(self.angle, self.throttle)
                        #     print("MID: ", mid, " Steering: ", 0.003*mid)
                        # else:
                        #     self.setDynThrottle(int(detection.spatialCoordinates.z))
                        #     self.vesc.run(0.5, self.throttle)
                    else:
                        self.stop()
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

                # cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
                # cv2.imshow("depth", depthFrameColor)
                # cv2.imshow("preview", frame)

                if cv2.waitKey(1) == ord('q'):
                    break