import cv2
import numpy as np

# parameters
ourClasses = ['car', 'motorbike', 'bus', 'truck']
confThreshold = 0.45
nmsThreshold = 0.50
inpWidth = 416
inpHeight = 416

# Load names of classes and turn that into a list
classesFile = "coco.names"

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Model configuration
modelConf = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

# Set up the net

net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def postProcess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold and classes[int(classID)] in ourClasses:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(centerX - width / 2)
                top = int(centerY - height / 2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    return [boxes[idx[0]] for idx in indices]


def vehicle_detection(frame):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    return postProcess(frame, outs)
