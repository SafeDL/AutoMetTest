import ImageReward as RM
import torch
import os
import requests
import io
import base64
from PIL import Image
import time
import numpy as np
from analyze_quality_metrics import calculate_psnr_and_ssim
import shutil
import clip

def get_ranking_images(reference_iamge, mutated_images, prompt):
    overall_scores = []
    for i, generated_image in enumerate(mutated_images):
        psnr_value, ssim_value = calculate_psnr_and_ssim(reference_iamge, generated_image)
        with torch.no_grad():
            aigc_value = IR_model.score(prompt, generated_image)
            similarity = compute_clip_similarity(generated_image, prompt)
       
        # Entropy weight method weights
        weights = [0.4983, 0.2947, 0.1064, 0.1006]

        score = sum([v * w for v, w in zip([psnr_value,ssim_value,aigc_value,similarity], weights)])
        overall_scores.append(score)

    # Calculate the overall score of SSIM and AIGC, and select the best_image with the highest score
    best_image = mutated_images[np.argmax(overall_scores)]
    best_score = overall_scores[np.argmax(overall_scores)]

    return best_image, best_score


def load_mutate_conditions(file_path):
    """Read prompts.txt file"""
    mutate_conditions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            condition = line.strip()
            if condition:
                mutate_conditions.append(condition)

    return mutate_conditions

def compute_clip_similarity(image, text_prompt):
    image = preprocess(image.convert("RGB")).unsqueeze(0).to("cuda")
    text = clip.tokenize([text_prompt]).to("cuda")

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()
    return similarity

def get_payload(target_dir, image_paths, mask_paths, prompt, clean_prompt):
    """
    :param target_dir: Folder to store mutated images
    :param image_paths: Paths to the original images to be mutated
    :param mask_paths: Masks corresponding to image_paths
    :param prompt: Mutation environment prompt for diffusion model test case generation
    :param clean_prompt: Clean prompt for the mutated environment, used for CLIP semantic alignment analysis
    """
    for kdx in range(0, len(image_paths)):
        with open(image_paths[kdx], "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
        with open(mask_paths[kdx], "rb") as mask_file:
            mask_base64 = base64.b64encode(mask_file.read()).decode('utf-8')
        payload = {
            "prompt": prompt,
            "negative_prompt": "worst quality,bad proportions,out of focus,lowres,blurry,low quality,",
            "styles": ["string"],
            "seed": -1,
            "steps": 20,
            "width": 640,
            "height": 384,
            "batch_size": 4,
            "cfg_scale": 8,
            "sampler_name": "DPM++ 2M SDE",
            "denoising_strength": 0.75,
            "init_images": [base64_encoded],  
            "model_id": "xxmix9realistic_v40",
            "save_images": False,
            "do_not_save_samples": False,
            "do_not_save_grid": False,
            "CLIP_stop_at_last_layers": 2,
            # The following specifies the parameters for ControlNet
            "alwayson_scripts": {
                "controlnet":
                    {
                        "args": [
                            {
                                "enabled": True,  
                                "control_mode": 'Balanced', 
                                "model": "control_v11p_sd15_seg_fp16 [ab613144]",
                                "module": "segmentation", 
                                "weight": 1.0,  
                                "guidance_start": 0, 
                                "guidance_end": 1.0, 
                                "pixel_perfect": True,  
                                "save_detected_map": False,
                                "image": base64_encoded, 
                            },
                            {
                                "enabled": True,  
                                "control_mode": 'Balanced',  
                                "model": "control_v11p_sd15_canny_fp16 [b18e0966]",  
                                "module": "canny",  
                                "weight": 1.0, 
                                "threshold_a": 100, 
                                "threshold_b": 200,  
                                "guidance_start": 0,  
                                "guidance_end": 1.0,  
                                "pixel_perfect": True,  
                                "processor_res": 512, 
                                "save_detected_map": False,
                                "image": base64_encoded, 
                            },
                        ]
                    },
                # The following specifies the parameters for Refiner
                "Refiner":
                    {
                        "args": [
                            True,  
                            "xxmix9realistic_v40.safetensors [18ed2b6c48]", 
                            0.75,  
                        ]
                    }
            },
        }
        r = call_image2image_api(payload)
        # The diffusion model generates multiple images in one batch, calculate the comprehensive score for each image, and save the one with the highest score
        reference_image = Image.open(image_paths[kdx])
        reference_image = reference_image.resize((640, 384))
        img_list = []
        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
            img_list.append(image)
        best_image, best_score = get_ranking_images(reference_iamge=reference_image, mutated_images=img_list, prompt=clean_prompt)

        best_image.save(os.path.join(target_dir, os.path.basename(image_paths[kdx])))


def call_image2image_api(payload):
    url = "http://127.0.0.1:7860"
    headers = {"Authorization": "Basic anptOmp6bTAzNDUxMQ=="}
    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload, headers=headers)
    r = response.json()
    return r


if __name__ == '__main__':
    start_time = time.time()

    # step1: Find the original images that need to be mutated
    original_conditions = [1, 3, 45, 69, 70, 71, 72, 75, 76, 77, 80, 81, 82, 83, 85, 86, 87, 88,
                           89, 90, 95, 98, 115, 120, 123, 127, 130, 131, 133, 134, 136, 165]

    # step2: Read the target test conditions
    prompts_file = '../CausalAnalysis/result/filtered_prompts.txt'
    mutate_conditions = load_mutate_conditions(prompts_file)
    clean_prompts_file = '../CausalAnalysis/result/clean_prompts.txt'
    clean_prompts = load_mutate_conditions(clean_prompts_file)

    # step3: Mutate each image in the original environment to the target test condition
    IR_model = RM.load("ImageReward-v1.0")  # Load AIGC scoring model
    clip_model, preprocess = clip.load("ViT-B/32", "cuda") # Load CLIP model

    for idx in range(0, 8):  
        mutation_folder = f"mutation_{idx}"
        if not os.path.exists(mutation_folder):
            os.makedirs(mutation_folder)

        for original_condition in original_conditions:
            current_mutation_condition = os.path.join(mutation_folder, 'condition_%d' % original_condition)
            if not os.path.exists(current_mutation_condition):
                os.makedirs(current_mutation_condition)

            image_paths = []
            mask_paths = []

            name = 'condition_%d' % original_condition
            real_images_folder = os.path.join('../Autopilot/driving_dataset/carla_collect', name, 'images')
            generated_images_folder = os.path.join('mutation_%d' % idx, name)

            for jdx in range(0, 200):
                if not os.path.exists(os.path.join(current_mutation_condition, f"{jdx:04d}.jpg")):
                    image_paths.append(os.path.join(real_images_folder, f"{jdx:04d}.jpg"))
                    mask_paths.append(os.path.join('../Autopilot/driving_dataset/carla_collect',
                                                   'condition_%d' % original_condition, 'masks', f"{jdx:04d}.jpg"))

            while image_paths:
                if not os.path.exists(os.path.join('./temp', generated_images_folder)):
                    os.makedirs(os.path.join('./temp', generated_images_folder))

                get_payload(target_dir=os.path.join('./temp',generated_images_folder), image_paths=image_paths, mask_paths=mask_paths,
                            prompt=mutate_conditions[idx], clean_prompt=clean_prompts[idx])

                if os.listdir('./temp'):
                    print(f"move {len(os.listdir('./temp'))} files to {current_mutation_condition}")
                else:
                    continue
                for file in os.listdir('./temp'):
                    shutil.move(os.path.join('./temp', file), current_mutation_condition)

        end_time = time.time()
        print(f"Mutation {idx} finished, time cost: {end_time - start_time:.2f}s")
