import numpy as np
from PIL import Image
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

class LeffaPredictor:
    def __init__(self):
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )
        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )
        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )
        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint, preprocess_garment):
        src_image = Image.open(src_image_path).convert("RGB")
        src_image = resize_and_center(src_image, 768, 1024)

        if preprocess_garment:
            ref_image = preprocess_garment_image(ref_image_path)
        else:
            ref_image = Image.open(ref_image_path).convert("RGB")
            ref_image = resize_and_center(ref_image, 768, 1024)

        model_parse, _ = self.parsing(src_image.resize((384, 512)))
        keypoints = self.openpose(src_image.resize((384, 512)))
        mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type).resize((768, 1024))

        src_image_array = np.array(src_image)
        densepose = Image.fromarray(self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1])

        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)

        output = self.vt_inference_hd(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )
        return output["generated_image"][0], mask, densepose
