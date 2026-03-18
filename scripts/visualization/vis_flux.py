import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from diffusers import FluxPipeline,FluxTransformer2DModel
from torch.utils.data import Dataset, DistributedSampler
import argparse

class PromptDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.prompts = [line.strip() for line in f if line.strip()][:1000]
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def sanitize_filename(text, max_length=200):
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', text)
    return sanitized[:max_length].rstrip() or "untitled"

def distributed_setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = distributed_setup()
    
    pipe = FluxPipeline.from_pretrained(
        "./data/flux",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    ).to("cuda")
    ft = FluxTransformer2DModel.from_pretrained(args.ft_dir, torch_dtype=torch.bfloat16,local_files_only=True).to(f"cuda:{local_rank}")
    pipe.transformer = ft
    dataset = PromptDataset(args.prompt_file)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # output_dir = Path(f"./assets/flux_visualization/grpo-100-1000")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in sampler:
        prompt = dataset[idx]
        try:
            generator = torch.Generator(device=f"cuda:{local_rank}")
            # generator.manual_seed(42 + idx + rank*1000)
            generator.manual_seed(42 + idx)
            # generator.manual_seed(42)
            image = pipe(
                prompt,
                guidance_scale=3.5,
                height=1024,
                width=1024,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=generator
            ).images[0]

            filename = sanitize_filename(prompt)
            save_path = output_dir / f"{filename}.jpg"
            
            counter = 1
            while save_path.exists():
                save_path = output_dir / f"{filename}_{counter}.jpg"
                counter += 1

            image.save(save_path)
            print(f"[Rank {rank}] Generated: {save_path.name}")

        except Exception as e:
            print(f"[Rank {rank}] Error processing '{prompt[:20]}...': {str(e)}")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_dir", type=str, default="data/flux/transformer", help="Fine-tuned transformer checkpoint directory.")
    parser.add_argument("--output_dir", type=str, default="outputs/visualization/flux", help="Directory to save generated images.")
    parser.add_argument("--prompt_file", type=str, default="scripts/visualization/prompt.txt")
    args = parser.parse_args()
    
    main()
