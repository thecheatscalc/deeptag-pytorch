"""
Minimal DeepTag wrapper class to expose model loading and a predict() hook.

This is a light wrapper; the detection/decoding logic remains in stag_decode.* modules.
"""

import numpy as np

class DeepTag:
    def __init__(self, tag_family: str = "aruco", device: str | None = None, model_dir: str | None = None):
        from deeptag_pytorch.settings import load_deeptag_models
        self.model_detector, self.model_decoder, self.device, self.tag_type, self.grid_size_cand_list = load_deeptag_models(tag_family, device=device, model_dir=model_dir)

    def predict(self, image: np.ndarray):
        """
        Run detection on an image (BGR numpy array). Returns decoded tags via DetectionEngine.
        """
        from deeptag_pytorch.stag_decode.detection_engine import DetectionEngine
        stag_image_processor = DetectionEngine(
            self.model_detector, self.model_decoder, self.device, self.tag_type, self.grid_size_cand_list,
            stg2_iter_num=2, min_center_score=0.2, min_corner_score=0.2, batch_size_stg2=4,
            hamming_dist=0, cameraMatrix=None, distCoeffs=None, codebook={}, tag_real_size_in_meter_dict={-1:0.1}
        )
        decoded = stag_image_processor.process(image, detect_scale=None)
        return decoded
