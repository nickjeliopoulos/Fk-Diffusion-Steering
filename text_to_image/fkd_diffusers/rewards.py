import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import clip
### IMPORTANT - from hpsv2x package, NOT original hpsv2 package
import hpsv2
import io
import numpy as np
import math

from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from diffusers.utils import numpy_to_pil, pt_to_pil

from .image_reward_utils import rm_load
from .llm_grading import LLMGrader

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
}


# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score"):
    # if reward_name != "LLMGrader":
        # print("`metric_to_chase` will be ignored as it only applies to 'LLMGrader' as the `reward_name`")
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)
    
    elif reward_name in ["Clip-Score","Clip-Score-only"]:
        return do_clip_score(images=images, prompts=prompts)
    
    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    elif reward_name == "LLMGrader":
        return do_llm_grading(images=images, prompts=prompts, metric_to_chase=metric_to_chase)

    elif reward_name == "JPEG_SCORE":
        return do_jpeg_score(images=images)

    elif reward_name == "JPEG_RAW":
        return do_jpeg(images=images)
    
    else:
        raise ValueError(f"Unknown metric: {reward_name}")
    
# JPEG Compression
def do_jpeg_score(*,images):
    scores = []
    for pil_img in images:
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=95)
        size = buffer.tell() / 1000.0
        score = ( 200.0 - size ) / 100.0
        score = max(score, 0.0) 
        buffer.close()
        scores.append(score)
    return scores

# JPEG Compression
def do_jpeg(*,images):
    sizes = []
    for pil_img in images:
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=95)
        size = buffer.tell() / 1000.0
        buffer.close()
        sizes.append(size)
    return sizes

# Compute human preference score
### NOTE: Changed from v2.1 to v2.0 for comparison with other methods
def do_human_preference_score(*, images, prompts, use_paths=False):
    if use_paths:
        scores = hpsv2.score(images, prompts, hps_version="v2.0")
        scores = [float(score) for score in scores]
    else:
        scores = []
        for i, image in enumerate(images):
            score = hpsv2.score(image, prompts[i], hps_version="v2.0")
            # print(f"Human preference score for image {i}: {score}")
            score = float(score[0])
            scores.append(score)

    # print(f"Human preference scores: {scores}")
    return scores

# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        arr_clip_result = []
        arr_img_features = []
        for i, prompt in enumerate(prompts):
            clip_result, feature_vect = REWARDS_DICT["Clip-Score"].score(
                prompt, images[i], return_feature=True
            )

            arr_clip_result.append(clip_result.item())
            arr_img_features.append(feature_vect['image'])

    # calculate diversity by computing pairwise similarity between image features
    diversity = torch.zeros(len(images), len(images))
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diversity[i, j] = (arr_img_features[i] - arr_img_features[j]).pow(2).sum()
            diversity[j, i] = diversity[i, j]
    n_samples = len(images)
    diversity = diversity.sum() / (n_samples * (n_samples - 1))

    return arr_clip_result, diversity.item()

# Compute ImageReward
def do_image_reward(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    with torch.no_grad():
        image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
        # image_reward_result = [REWARDS_DICT["ImageReward"].score(prompt, images[i]) for i, prompt in enumerate(prompts)]

    return image_reward_result

# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        ### ORIGINAL
        # REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
        ### MODIFIED for EvoAlgs, CVPR26
        REWARDS_DICT["Clip-Score"] = ModifiedCLIPScore(device="cuda:0", inference_dtype=torch.float16)
    with torch.no_grad():
        clip_result = [
            REWARDS_DICT["Clip-Score"].score(prompt, images[i]).detach().cpu().item()
            for i, prompt in enumerate(prompts)
        ]
    return clip_result


# Compute LLM-grading
def do_llm_grading(*, images, prompts, metric_to_chase="overall_score"):
    global REWARDS_DICT
    
    if REWARDS_DICT["LLMGrader"] is None:
        REWARDS_DICT["LLMGrader"]  = LLMGrader()
    llm_grading_result = [
        REWARDS_DICT["LLMGrader"].score(images=images[i], prompts=prompt, metric_to_chase=metric_to_chase)
        for i, prompt in enumerate(prompts)
    ]
    return llm_grading_result


'''
@File       :   CLIPScore.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore.
* Based on CLIP code base
* https://github.com/openai/CLIP
'''


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # if (type(image_path).__name__=='list'):
        #     _, rewards = self.inference_rank(prompt, image_path)
        #     return rewards

        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))

        # score
        rewards = torch.sum(
            torch.mul(txt_features, image_features), dim=1, keepdim=True
        )

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}

        return rewards.detach().cpu().numpy().item()

### Ripped from DiffusionEvoAlgs (also used in Fk-Diffusion-Steering fork)
### Really ought to change the preprocess code, however I am keeping it "as-is" because this is consistent across survyed methods and codebases
class ModifiedCLIPScore(torch.nn.Module):
    def __init__(self, device, inference_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.clip_model_name = "openai/clip-vit-base-patch16"
        self.dtype = inference_dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(
            self.clip_model_name, 
            torch_dtype=self.dtype
        ).eval().to(device=device)
        
    def score(self, prompts, images, return_feature=False):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            pil_images = [Image.fromarray(image) for image in images]

        elif isinstance(images, np.ndarray):
            pil_images = numpy_to_pil(images)

        elif isinstance(images, list):
            if not isinstance(images[0], Image.Image):
                raise ValueError(f"Images must contain PIL Images if it is a List - instead it contains {type(images[0])}")
            pil_images = [image for image in images]
        
        elif isinstance(images, Image.Image):
            pil_images = [images]

        else:
            raise ValueError(f"Images type {type(images)} unsupported")

        if isinstance(prompts, list):
            prompt = prompts[0]
        elif isinstance(prompts, str):
            prompt = prompts
        else:
            raise ValueError(f"Prompts of type {type(prompts)} invalid")

        ### Man I hate this
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device=self.device)
        outputs = self.clip_model(**inputs)
        score = outputs[0][:, 0]

        if return_feature:
           raise ValueError("No support for return_feature.")

        return score