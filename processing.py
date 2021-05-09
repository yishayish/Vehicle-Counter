import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def show_bbox(bbox1, bbox2, win='test'):
    frame = np.zeros((800, 600, 3))
    for i, b in enumerate(bbox1):
        b = b[0]
        print(frame.shape, b)
        cv2.rectangle(frame, b, [100, 255, 0], 2)
        cv2.putText(frame, str(i), (b[0] - 10, b[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 20, 150), 1)
    for i, b in enumerate(bbox2):
        cv2.rectangle(frame, b, [0, 255, 255], 2)
        cv2.putText(frame, str(i), (b[0] - 10, b[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 20, 250), 1)
    cv2.imshow(win, frame)
    cv2.waitKey()


THRESHOLD = 0.05

"""
# counter


# New detections
    threshold of IOU
    objects new
    objects left



# update objects:
    IOU intersection over union N x N
    linear assignment - hungarian algorithm


"""
counter = 0
objects = []


def intersection(rect, other):
    # rect -> [left, top, width, height]
    dx = min(rect[0] + rect[2], other[0] + other[2]) - max(rect[0], other[0])
    dy = min(rect[1] + rect[3], other[1] + other[3]) - max(rect[1], other[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0


def area(rect):
    return rect[2] * rect[3]


def IOU(rect, other, thresh=THRESHOLD):
    i = intersection(rect, other)
    val = i / (area(rect) + area(other) - i)
    return val if val > thresh else 0


def IOU_matrix(old_bbox, new_bbox):
    # show_bbox(old_bbox, new_bbox)
    N, M = len(new_bbox), len(old_bbox)
    mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            mat[i, j] = IOU(new_bbox[i], old_bbox[j][0])
    # print(mat)
    return mat


def get_new_objects(mat):
    n = np.where(~mat.any(axis=1))[0]
    return set(n)


def init(bboxs):
    global counter
    for i in bboxs:
        objects.append((i, counter))
        counter += 1
    return objects


def process_New_detections(detections):
    global counter, objects

    if not objects:
        return init(detections)

    matIOU = IOU_matrix(objects, detections)
    news = get_new_objects(matIOU)
    row_ind, col_ind = linear_sum_assignment(-matIOU)

    objects_updated = []
    for idx, i in enumerate(row_ind):
        if i in news:
            objects_updated.append((detections[i], counter))
            counter += 1
        else:
            objects_updated.append((detections[i], objects[col_ind[idx]][1]))
    objects = objects_updated
    objects.sort(key=lambda x: x[1])
    return objects


def update_objects(upobjs):
    global objects
    objects = upobjs
