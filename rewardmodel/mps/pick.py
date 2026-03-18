# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import cv2
import random


class pick():
    def __init__(self, device, processor_name_or_path, model_pretrained_name_or_path):

        # load model
        self.device = device

        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(self.device)


    def calc_probs(self, prompt, video_path, device):
        self.model.to(device)

        def _extract_frames(video_path):
            # 打开视频文件
            video_capture = cv2.VideoCapture(video_path)
            
            if not video_capture.isOpened():
                print("Error: Cannot open video.")
                return
            
            # 获取视频的总帧数
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 随机抽取帧的索引
            # frame_indices = random.sample(range(total_frames), num_frames)
            
            frames = []
            
            for idx in range(total_frames):
                # 设置当前帧的位置
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
                
                # 读取帧
                ret, frame = video_capture.read()
                if ret:
                    # 将 BGR 转换为 RGB，Pillow 使用的是 RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                else:
                    print(f"Error: Cannot read frame at index {idx}")
            
            video_capture.release()
            return frames
        # preprocess
        images = _extract_frames(video_path)
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)


        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            # probs = torch.softmax(scores, dim=-1)
        # breakpoint()
        self.model.to('cpu')
        return scores.mean()

# pil_images = [Image.open("ship.png"), Image.open("shipbad.png")]
# prompt = "A small boat is bravely battling the waves, forging ahead. The vast blue sea is tumultuous, with white spray crashing against the hull, but the little boat shows no fear, steadfastly sailing towards the distant horizon. Sunlight sprinkles across the water's surface, shimmering with golden hues, adding a touch of warmth to this magnificent scene. As the camera zooms in, one can see the flag on board fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. This scene, full of power, is inspiring and uplifting, showcasing the fearlessness and perseverance when facing challenges."
# print(calc_probs(prompt, pil_images))