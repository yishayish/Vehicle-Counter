import cv2

from tqdm import tqdm


def crop_frame(frame, cut):
    return frame[int(cut[1]):int(cut[1] + cut[3]), int(cut[0]):int(cut[0] + cut[2])]


def paint_roi(frame, cut):
    roi = frame[int(cut[1]):int(cut[1] + cut[3]), int(cut[0]):int(cut[0] + cut[2])]
    roi[:, :, 0] = 0
    frame[int(cut[1]):int(cut[1] + cut[3]), int(cut[0]):int(cut[0] + cut[2])] = roi


def read(file, queue, length, start_from=0):
    bar = tqdm(total=length, desc='Reader Process')
    c = 0
    reader = cv2.VideoCapture(file)
    reader.set(1, start_from)
    while True:
        if not queue.full():
            ok, frame = reader.read()
            if not ok:
                return
            queue.put(frame)
            c += 1
            bar.update(1)
            if c == length:
                print('\nReader process terminated')
                return


def write(queue, length=0, screen=True, winName='', file=None, size=(640, 480), fps=10, screen_size=(800, 600)):
    if not screen and not file:
        raise ValueError('ERROR: No output option set')
    if file:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file, fourcc, fps, size)
    if screen:
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, *screen_size)
    bar = tqdm(total=length, desc='Writer Process')
    c = 0
    while True:
        if not queue.empty():
            frame = queue.get()
            if screen:
                frame = cv2.resize(frame, screen_size)
                cv2.imshow(winName, frame)
                cv2.waitKey(1000 // fps)
            if file:
                frame = cv2.resize(frame, size)
                out.write(frame)
            c += 1
            bar.update(1)
            if c == length:
                out.release()
                print('\nWriter process terminated')
                return


def draw_bounding_box(frame, rectangles, num, crop=False, size=None):
    if crop:
        paint_roi(frame, size)
    for object_, id_ in rectangles:
        if object_:
            a, b, c, d = object_
            if crop:
                a += size[0]
                b += size[1]
            cv2.rectangle(frame, (a, b, c, d), [100, 255, 0], 2)
            cv2.putText(frame, str(id_), (a - 10, b - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (32, 20, 150), 1)
    cv2.putText(frame, str(num), (0, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# q = Queue(maxsize=100)
# t = threading.Thread(target=read, args=('V1.mp4', q))
# t.start()
# write(q, fps=FPS)
