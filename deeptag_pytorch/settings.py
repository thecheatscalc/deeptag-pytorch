"""
Load / initialize DeepTag detector and decoder models.

Converted from the repository's top-level deeptag_model_setting.py and adapted to package layout.
"""

import os
import torch
from pathlib import Path
import importlib.resources as pkg_resources

def _resolve_models_dir(model_dir: str | None = None) -> str:
    if model_dir:
        return model_dir
    try:
        pkg_root = pkg_resources.files("deeptag_pytorch")
        models_path = pkg_root.joinpath("models")
        return str(models_path)
    except Exception:
        here = Path(__file__).resolve().parent
        return str((here.parent / "models").resolve())

def load_deeptag_models(tag_family, device=None, model_dir=None):
    # relative imports for package layout
    from .network.decoder_net import DecoderNet as decoder
    from .network.detector_net import DetectorNet as detector

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TAG_TYPE = 1
    if tag_family in ('aruco', 'apriltag', 'artookkitplus'):
        TAG_TYPE = 1
    elif tag_family == 'topotag':
        TAG_TYPE = 0
    elif tag_family in ('apriltagxo', 'apriltagxa'):
        TAG_TYPE = 3
    elif tag_family == 'runetag':
        TAG_TYPE = 4

    checkpoint_dir = _resolve_models_dir(model_dir)
    checkpoint_dir_2stg = checkpoint_dir

    num_classes_keypoints = 2
    if TAG_TYPE == 0:
        model_filename = 'topotag_roi_detector.pth'
        model_filename_2stg = 'topotag_decoder.pth'
        tag_type = 'topotag'
        grid_size_cand_list = [4]
        num_classes = 4
    elif TAG_TYPE == 1:
        model_filename = 'arucotag_roi_detector.pth'
        model_filename_2stg = 'arucotag_decoder.pth'
        tag_type = 'arucotag'
        grid_size_cand_list = [4, 5, 6, 7]
        num_classes = 3
    elif TAG_TYPE == 3:
        model_filename = 'arucotag_xab_roi_detector.pth'
        model_filename_2stg = 'arucotag_xab_decoder.pth'
        tag_type = 'arucotag'
        grid_size_cand_list = [6]
        num_classes = 3
    elif TAG_TYPE == 4:
        model_filename = 'runetag_roi_detector.pth'
        model_filename_2stg = 'runetag_decoder_512x.pth'
        tag_type = 'runetag'
        grid_size_cand_list = []
        num_classes = 3
        num_classes_keypoints = 3

    print('===========> loading model <===========')
    num_masks = 2
    num_channels = 128
    num_channels_refiner = 32

    model_detector = detector(num_channels=num_channels, num_masks=num_masks, num_classes=2, num_classes_keypoints=num_classes_keypoints)
    detector_path = os.path.join(checkpoint_dir, model_filename)
    state_dict = torch.load(detector_path, map_location=device)
    model_detector.load_state_dict(state_dict)
    model_detector.to(device)

    model_decoder = decoder(num_channels=num_channels_refiner, num_masks=num_masks, num_classes=num_classes)
    decoder_path = os.path.join(checkpoint_dir_2stg, model_filename_2stg)
    state_dict_2stg = torch.load(decoder_path, map_location=device)
    model_decoder.load_state_dict(state_dict_2stg)
    model_decoder.to(device)

    return model_detector, model_decoder, device, tag_type, grid_size_cand_list

def load_model_settings():
    try:
        with pkg_resources.open_text("deeptag_pytorch", "config_image.json") as fh:
            import json
            return json.load(fh)
    except Exception:
        return {}
def load_video_settings():
    try:
        with pkg_resources.open_text("deeptag_pytorch", "config_video.json") as fh:
            import json
            return json.load(fh)
    except Exception:
        return {}
