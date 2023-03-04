from insightface.app import FaceAnalysis
import time


class FaceAnalyzer:
    def __init__(self, models_dir='../models/', **kwargs):
        self.model = FaceAnalysis(name='antelopev2', root=models_dir,
                                  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def analyze(self, img, max_num=0):
        start_time = time.time()
        faces = self.model.get(img, max_num)
        print("--- %s seconds ---" % (time.time() - start_time))
        return faces
