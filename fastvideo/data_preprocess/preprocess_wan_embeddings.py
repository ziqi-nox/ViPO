import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from wan.modules.t5 import T5EncoderModel
from wan.configs import t2v_1_3B

def preprocess_text_data(args):
    """预处理文本数据，将文本转换为T5编码并保存为npy文件"""
    
    # 加载原始JSON数据
    print(f"Loading data from {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果数据是字典格式，转换为列表
    if isinstance(data, dict):
        data_list = list(data.values())
    else:
        data_list = data
    
    print(f"Found {len(data_list)} items")
    
    # 初始化T5编码器
    config = t2v_1_3B
    print("Loading T5 encoder...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=torch.float32,
        device=device,
        checkpoint_path=os.path.join(args.wan_model_path, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.wan_model_path, config.t5_tokenizer),
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    new_data = []
    
    print("Processing texts...")
    for i in tqdm(range(len(data_list)), desc="Encoding texts"):
        item = data_list[i]
        
        # 提取文本
        if isinstance(item, dict):
            caption = item.get('caption', item.get('text', 'A video'))
        else:
            caption = str(item)
        
        # 使用T5编码文本
        with torch.no_grad():
            context = text_encoder([caption], device=device)
            
            # 检查context的类型并处理
            if isinstance(context, list):
                # 如果是list，取第一个元素并转换为numpy
                context_tensor = context[0].cpu().numpy()
            else:
                # 如果是tensor，直接转换为numpy
                context_tensor = context.cpu().numpy()
            
            # 保存为npy文件
            npy_filename = f"context_{i:06d}.npy"
            npy_path = os.path.join(args.output_dir, npy_filename)
            
            np.save(npy_path, context_tensor)
        
        # 只保存必要信息
        new_data.append({
            'caption': caption,
            'context_path': npy_path
        })
        
        # 定期清理显存
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
    
    # 保存简化的JSON文件
    output_json = os.path.join(args.output_dir, 'processed_wan_prompt.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessed {len(new_data)} items")
    print(f"Text embeddings saved to: {args.output_dir}")
    print(f"New JSON saved to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="输入的原始JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--wan_model_path", type=str, required=True, help="WAN模型路径")
    
    args = parser.parse_args()
    preprocess_text_data(args)