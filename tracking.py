import cv2

trackers = []


def track_objects_init(frame, objects):
    trackers.clear()
    for box, id in objects:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, tuple(box))
        trackers.append((tracker, id))


def update_trackers(frame):
    ret = []
    for t, id in trackers:
        ok, b = t.update(frame)
        bbox = [int(i) for i in b] if ok else None
        ret.append((bbox, id))
    return ret
