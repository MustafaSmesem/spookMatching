import cv2
from insightface.app import FaceAnalysis
import time

if __name__ == '__main__':
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    vid = cv2.VideoCapture(0)
    frameNumber = 0
    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        if not ret:
            print('ret is none')
            break
        if frame is None:
            print('frame is empty')
            break

        if frameNumber % 3 == 0:
            start = time.time()
            faces = app.get(frame)
            # rimg = app.draw_on(frame, faces)
            print(f'detection [%d]: %.3f sn' % (frameNumber, time.time() - start))

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frameNumber += 1

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
