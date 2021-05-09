import detection
import processing
import tracking
from IO import *


def pipeline(in_queue, out_queue, skip_frames=10, length=0, crop=False, size=None):
    bar = tqdm(total=250, desc='Main Process')
    c = 0
    while True:
        if not in_queue.empty() and not out_queue.full():
            frame = in_queue.get()
            if crop:
                original_frame = frame
                frame = crop_frame(frame, size)
            if not c % skip_frames:
                vehicles = detection.vehicle_detection(frame)
                vehicles = processing.process_New_detections(vehicles)
                tracking.track_objects_init(frame, vehicles)

            else:
                vehicles = tracking.update_trackers(frame)
                processing.update_objects(vehicles)

            if crop:
                draw_bounding_box(original_frame, vehicles, c, crop=crop, size=size)
                out_queue.put(original_frame)
            else:
                draw_bounding_box(frame, vehicles, c)
                out_queue.put(frame)
            c += 1
            bar.update(1)
            if c == length:
                print('\nMain process terminated')
                return
