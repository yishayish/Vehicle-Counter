from multiprocessing import Process, Queue
from time import sleep

import cv2

from IO import read, write
from main import pipeline


def get_parameters(input_path):
    v = cv2.VideoCapture(input_path)
    total_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    _, frame = v.read()
    desc = 'Select a ROI and then press SPACE or ENTER button!'
    cut = cv2.selectROI(desc, frame, showCrosshair=False)
    v.release()
    cv2.destroyAllWindows()
    return total_frames, cut


input_path = 'V1.mp4'
output_path = 'result.avi'
screen_display = True
length = 400
debug = False
cut = 0

if __name__ == '__main__':
    total_frames, cut = get_parameters(input_path)
    if not length:
        length = total_frames

    winName = 'Vehicle Counter'

    first_Queue = Queue(maxsize=300)
    second_Queue = Queue(maxsize=300)
    p_read = Process(target=read, args=(input_path, first_Queue, length),
                     kwargs={'start_from': 0})
    p_main = Process(target=pipeline, args=(first_Queue, second_Queue),
                     kwargs={'length': length, 'crop': True, 'size': cut})
    p_write = Process(target=write, args=(second_Queue,),
                      kwargs={'screen': screen_display, 'file': output_path,
                              'length': length, 'winName': winName, 'screen_size': (800, 600)})

    p_read.start()

    if not debug:
        p_main.start()
        sleep(7)
        p_write.start()
    else:
        p_write.start()
        pipeline(first_Queue, second_Queue, length=length, crop=True, size=cut)
