import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs

# step 9 
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

# follow at https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/
# step one
pipeline = depthai.Pipeline()

# step 2
cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

# step 3
detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
# Set path of the blob (NN model). We will use blobconverter to convert&download the model
#detection_nn.setBlobPath("/mobilenet-ssd_openvino_2021.4_6shave.blob")
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setConfidenceThreshold(0.5)

# step 4
cam_rgb.preview.link(detection_nn.input)

# step 5
xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Define sources and outputs
monoLeft = pipeline.create(depthai.node.MonoCamera)
monoRight = pipeline.create(depthai.node.MonoCamera)
depth = pipeline.create(depthai.node.StereoDepth)
xout = pipeline.create(depthai.node.XLinkOut)

xout.setStreamName("disparity")

# Properties
monoLeft.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(depthai.CameraBoardSocket.LEFT)
monoRight.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(depthai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# Create a colormap
colormap = pipeline.create(depthai.node.ImageManip)
colormap.initialConfig.setColormap(depthai.Colormap.STEREO_TURBO, depth.initialConfig.getMaxDisparity())
colormap.initialConfig.setFrameType(depthai.ImgFrame.Type.NV12)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(colormap.inputImage)
colormap.out.link(xout.input)

# step 6
with depthai.Device(pipeline) as device:
    # step 7
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    # Output queue will be used to get the disparity frames from the outputs defined above
    q_depth = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    
    # step 8
    frame = None
    detections = []

    # step 10
    while True:
        # step 11
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        in_disparity = q_depth.get()

        #step 12
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
        # step 13
        if in_nn is not None:
            detections = in_nn.detections
        if in_disparity is not None:
            depths = in_disparity.getCvFrame()
        
        print(np.shape(depths))
        #depths = depths[200:500,20:320]
        cv2.circle(depths, (285,215), 2, (255,255,255), 2)
        cv2.circle(depths, (285,265), 2, (255,255,255), 2)

        cv2.imshow("disparity", depths)
        #step 14
        if frame is not None:
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.circle(frame, (150,150), 2, (255,255,255), 2)
                cv2.circle(frame, (150,200), 2, (255,255,255), 2)

        #         if (detection.label == 15):
        #             print("I found a... human...")
        #         else:
        #             print("Not a ... human ...")
        #     cv2.imshow("preview", frame)

                if (detection.label == 15):
                    print("I found a... human...")
                    print(bbox[0], bbox[2])
                    mid = int((bbox[2]-bbox[0])/2)
                    print(mid)
                    
                cv2.imshow("preview", frame)
        #step 15
        if cv2.waitKey(1) == ord('q'):
            break