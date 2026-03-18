# import
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
# breakpoint()
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from transformers import CLIPFeatureExtractor, CLIPImageProcessor, AutoTokenizer
from io import BytesIO
import trainer
import trainer.models
import cv2
import random
from torch import nn, einsum



class mps():
    def __init__(self, device, processor_name_or_path, model_ckpt_path):
        # load model
        self.device = device
        self.image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)

        self.model = torch.load(model_ckpt_path,weights_only=False)
        self.model.model.config.text_config._attn_implementation_internal = "flash_attention_2"
        self.model.model.config.vision_config._attn_implementation_internal = "flash_attention_2"
        self.model.model.config.text_config.eos_token_id = 2
        self.model.eval().to(device)

    def infer_one_sample(self, image, prompt, device, condition=None):
        def _process_image(image):
            if isinstance(image, dict):
                image = image["bytes"]
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            if isinstance(image, str):
                image = Image.open( image )
            image = image.convert("RGB")
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
            return pixel_values
        
        def _tokenize(caption):
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            return input_ids
        
        image_input = _process_image(image).to(device)
        text_input = _tokenize(prompt).to(device)
        if condition is None:
            condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."
        condition_batch = _tokenize(condition).repeat(text_input.shape[0],1).to(device)

        with torch.no_grad():
            text_f, text_features = self.model.model.get_text_features(text_input)

            image_f = self.model.model.get_image_features(image_input.half())
            condition_f, _ = self.model.model.get_text_features(condition_batch)

            sim_text_condition = einsum('b i d, b j d -> b j i', text_f, condition_f)
            sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
            sim_text_condition = sim_text_condition / sim_text_condition.max()
            mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))
            mask = mask.repeat(1,image_f.shape[1],1)
            image_features = self.model.cross_model(image_f, text_f,mask.half())[:,0,:]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_score = self.model.logit_scale.exp() * text_features @ image_features.T

        return image_score[0]

    def infer_example(self, video_path, prompt, device):

        def _extract_frames(video_path):
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                print("Error: Cannot open video.")
                return
            # 获取视频的总帧数
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
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

        images = _extract_frames(video_path)

        scores = []
        self.model.to(device)
        for image in images:
            score = self.infer_one_sample(image, prompt, device)
            scores.append(score)
        scores = torch.stack(scores, dim=-1).mean()
        self.model.to('cpu')
        return scores
