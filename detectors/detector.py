class Detector:
    def __init__(self, config, checkpoint, detection_class, device='cuda:0') -> None:
        super().__init__()
        self.config = config
        self.checkpoint = checkpoint
        self.detection_class = detection_class
        self.device = device

    def infer(self, img, detection_threshold):
        raise NotImplementedError()
