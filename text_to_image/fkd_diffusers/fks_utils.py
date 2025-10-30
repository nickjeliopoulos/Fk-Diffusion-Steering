"""
Utility functions for the FKD pipeline.
"""
import torch
from diffusers import DDIMScheduler

from .fkd_pipeline_sdxl import FKDStableDiffusionXL
from .fkd_pipeline_sd import FKDStableDiffusion

from .rewards import (
    do_clip_score,
    do_clip_score_diversity,
    do_image_reward,
    do_human_preference_score,
    do_llm_grading,
    do_jpeg,
    do_jpeg_score
)


def get_model(model_name):
    """
    Get the FKD-supported model based on the model name.
    """
    if model_name == "stable-diffusion-xl":
        pipeline = FKDStableDiffusionXL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-5":
        pipeline = FKDStableDiffusion.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-4":
        pipeline = FKDStableDiffusion.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-2-1":
        pipeline = FKDStableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    return pipeline



def do_eval(*, prompt, images, metrics_to_compute):
    """
    Compute the metrics for the given images and prompt.
    """
    results = {}
    for metric in metrics_to_compute:
        if metric == "Clip-Score":
            results[metric] = {}
            (
                results[metric]["result"],
                results[metric]["diversity"],
            ) = do_clip_score_diversity(images=images, prompts=prompt)
            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "ImageReward":
            results[metric] = {}
            results[metric]["result"] = do_image_reward(images=images, prompts=prompt)

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "Clip-Score-only":
            results[metric] = {}
            results[metric]["result"] = do_clip_score(images=images, prompts=prompt)

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()
        elif metric == "HumanPreference":
            results[metric] = {}
            results[metric]["result"] = do_human_preference_score(
                images=images, prompts=prompt
            )

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "LLMGrader":
            results[metric] = {}
            out = do_llm_grading(images=images, prompts=prompt)
            print(out)
            results[metric]["result"] = out

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "JPEG_SCORE" or metric == "JPEG_RAW":
            results["JPEG_SCORE"] = {}

            results["JPEG_SCORE"]["result"] = do_jpeg_score(images=images)
            results_arr = torch.tensor(results["JPEG_SCORE"]["result"])

            results["JPEG_SCORE"]["mean"] = results_arr.mean().item()
            results["JPEG_SCORE"]["std"] = results_arr.std().item()
            results["JPEG_SCORE"]["max"] = results_arr.max().item()
            results["JPEG_SCORE"]["min"] = results_arr.min().item()

            results["JPEG_RAW"] = {}

            results["JPEG_RAW"]["result"] = do_jpeg(images=images)
            results_arr = torch.tensor(results["JPEG_RAW"]["result"])

            results["JPEG_RAW"]["mean"] = results_arr.mean().item()
            results["JPEG_RAW"]["std"] = results_arr.std().item()
            results["JPEG_RAW"]["max"] = results_arr.max().item()
            results["JPEG_RAW"]["min"] = results_arr.min().item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results
