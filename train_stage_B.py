import argparse
import os
import shutil
from dataclasses import dataclass

import numpy as np
import torch
import torchvision
import transformers
import wandb
import webdataset as wds
from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from transformers.utils import is_torch_bf16_available, is_torch_tf32_available
from warmup_scheduler import GradualWarmupScheduler
from webdataset.handlers import warn_and_continue

from modules import Paella, sample, EfficientNetEncoder, Wrapper
from utils import WebdatasetFilter, transforms, effnet_preprocess, identity, ImageTextDataset
from vqgan import VQModel

transformers.utils.logging.set_verbosity_error()

# PARAMETERS
parser = argparse.ArgumentParser()

parser.add_argument(
    "--updates",
    type=int,
    default=1500000,
    help="The amount of training steps",
)
parser.add_argument(
    "--warmup_updates",
    type=int,
    default=10000,
    help="The amount of warmup steps",
)
parser.add_argument(
    "--ema_start",
    type=int,
    default=5000,
    help="The amount of steps before starting EMA",
)
parser.add_argument(
    "--ema_every",
    type=int,
    default=100,
    help="The amount of steps inbetween saving EMA",
)
parser.add_argument(
    "--ema_beta",
    type=float,
    default=0.9,
    help="The beta to use for EMA",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="The batch size to use",
)
parser.add_argument(
    "--grad_accum_steps",
    type=int,
    default=1,
    help="The amount of gradient accumulation steps",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=5,
    help="The amount of steps inbetween printing",
)
parser.add_argument(
    "--extra_ckpt_every",
    type=int,
    default=10000,
    help="The amount of steps inbetween saving a ckpt",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="The learning rate",
)
parser.add_argument(
    "--generate_new_wandb_id",
    action="store_true",
    default=False,
    help="Whether to create a new wandb job or recycle one stored in the ckpt"
)
parser.add_argument(
    "--hf_dataset_name",
    type=str,
    default="",
    help="The hf dataset name",
)
parser.add_argument(
    "--wd_dataset_location",
    type=str,
    default="",
    help="The wd dataset location",
)
parser.add_argument(
    "--cache_path",
    type=str,
    default="",
    help="The cache path for hf datasets",
)
parser.add_argument(
    "--text_column",
    type=str,
    default="text",
    help="The column of the text data",
)
parser.add_argument(
    "--image_column",
    type=str,
    default="image",
    help="The column of the image data",
)
parser.add_argument(
    "--run_name",
    type=str,
    default="Würstchen-v4-512-CLIP-text",
    help="The wandb run name",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="output/würstchen/",
    help="The output path of eval images",
)
parser.add_argument(
    "--load",
    action="store_true",
    default=False,
    help="Whether to load a checkpoint"
)
parser.add_argument(
    "--load_checkpoint_path",
    type=str,
    default="",
    help="The path to the ckpt file you want to load",
)
parser.add_argument(
    "--save_checkpoint_path",
    type=str,
    default="",
    help="The path to where you want to save models",
)
parser.add_argument(
    "--vq_model_path",
    type=str,
    default="",
    help="The path to where your vq model",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="",
    help="The wandb project you wish to use",
)
parser.add_argument(
    "--wadnv_entity",
    type=str,
    default="",
    help="The wandb entity you wish to use",
)


@dataclass
class Arguments:
    updates = 1500000
    warmup_updates = 10000
    ema_start = 5000
    ema_every = 5000
    ema_beta = 0.9
    batch_size = 8
    grad_accum_steps = 1
    max_iters = updates * grad_accum_steps
    print_every = 5 * grad_accum_steps
    extra_ckpt_every = 10000
    lr = 1e-4
    generate_new_wandb_id = False

    hf_dataset_name = ""
    wd_dataset_location = ""
    cache_path = ""

    text_column = "text"
    image_column = "image"
    run_name = "Würstchen-v4-512-CLIP-text"
    output_path = f"output/würstchen/"
    load = True
    load_checkpoint_path = ""
    save_checkpoint_path = ""
    vq_model_path = ""

    wandb_project = ""
    wandb_entity = ""


args = parser.parse_args(namespace=Arguments())


def train(gpu_id):
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.save_checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_checkpoint_path, args.run_name), exist_ok=True)

    device = torch.device(gpu_id)

    # only ampere gpu architecture allows these
    _float16_dtype = torch.float16 if not is_torch_bf16_available() else torch.bfloat16
    if is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- PREPARE DATASET ---
    if args.wd_dataset_location:
        dataset = wds.WebDataset(
            args.wd_dataset_location, resampled=True, handler=warn_and_continue
        ).select(
            WebdatasetFilter(min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99)
        ).shuffle(690, handler=warn_and_continue).decode(
            "pilrgb", handler=warn_and_continue
        ).to_tuple(
            "jpg", "txt", handler=warn_and_continue
        ).map_tuple(
            transforms, identity, handler=warn_and_continue
        )
    else:
        dataset = load_dataset(
            args.hf_dataset_name,
            cache_dir=args.cache_path,
            save_infos=True,
            split="train", )
        dataset = ImageTextDataset(dataset, args.image_column, args.text_column)

    real_batch_size = args.batch_size // args.grad_accum_steps
    print("REAL BATCH SIZE:", real_batch_size)

    dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)

    # --- PREPARE MODELS ---
    try:
        checkpoint = torch.load(args.load_checkpoint_path, map_location=device) if os.path.exists(
            args.load_checkpoint_path) and args.load else None
    except RuntimeError as e:
        if os.path.exists(f"{args.load_checkpoint_path}.bak"):
            os.remove(args.load_checkpoint_path)
            shutil.copyfile(f"{args.load_checkpoint_path}.bak", args.load_checkpoint_path)
            if args.load:
                checkpoint = torch.load(args.load_checkpoint_path, map_location=device)
            else:
                checkpoint = None
        else:
            raise e

    # - vqmodel -
    vqmodel = VQModel().to(device)
    vqmodel.load_state_dict(torch.load(args.vq_model_path, map_location=device)['state_dict'])
    vqmodel.eval().requires_grad_(False)

    # - CLIP text encoder
    clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(
        device).eval().requires_grad_(False)
    clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    # - Paella Model as generator - 
    generator = Paella(byt5_embd=1024).to(device)
    if checkpoint is not None:
        generator.load_state_dict(checkpoint['state_dict'])

    # - EfficientNet -
    effnet = EfficientNetEncoder(effnet="efficientnet_v2_l").to(device)
    if checkpoint is not None:
        if "effnet_state_dict" in checkpoint:
            effnet.load_state_dict(checkpoint['effnet_state_dict'])

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Wrapper(effnet, generator, device=device).to(device))

    # - SETUP WANDB -
    print("Num trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if checkpoint is not None and not args.generate_new_wandb_id:
        try:
            run_id = checkpoint['wandb_run_id']
        except KeyError:
            run_id = wandb.util.generate_id()
    else:
        run_id = wandb.util.generate_id()
    wandb.init(project=args.wandb_project, name=args.run_name, entity=args.wandb_entity, id=run_id,
               resume="allow")

    # SETUP OPTIMIZER, SCHEDULER & CRITERION
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))  # eps=1e-4
    # optimizer = Lion(model.parameters(), lr=lr / 3) # eps=1e-4
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_updates)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Failed loading optimizer, skipping...")
        scheduler.last_epoch = checkpoint['scheduler_last_step']
    scaler = torch.cuda.amp.GradScaler()
    if checkpoint is not None and 'grad_scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

    start_iter = 1
    grad_norm = torch.tensor(0, device=device)
    if checkpoint is not None:
        start_iter = checkpoint['scheduler_last_step'] * args.grad_accum_steps + 1

    skipped = 0
    loss_adjusted = 0.

    if checkpoint is not None:
        del checkpoint  # cleanup memory
        torch.cuda.empty_cache()

        # -------------- START TRAINING --------------
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, args.max_iters + 1))
    model.train()
    for it in pbar:
        images, captions = next(dataloader_iterator)
        images = images.to(device)

        with torch.cuda.amp.autocast(dtype=_float16_dtype), torch.no_grad():
            if np.random.rand() < 0.05:  # 90% of the time, drop the CLIP text embeddings (indepentently)
                clip_captions = [''] * len(captions)  # 5% of the time drop all the captions
            else:
                clip_captions = captions
            clip_tokens = clip_tokenizer(clip_captions, truncation=True, padding="max_length",
                                         max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
            clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

            t = (1 - torch.rand(images.size(0), device=device)).mul(1.08).add(0.001).clamp(0.001, 1.0)
            latents = vqmodel.encode(images)[2]
            noised_latents, mask = model.module.generator.add_noise(latents, t)
            loss_weight = model.module.generator.get_loss_weight(t, mask)

            effnet_preproc = effnet_preprocess(images)

        with torch.cuda.amp.autocast(dtype=_float16_dtype):
            pred = model(noised_latents, t, effnet_preproc, clip_text_embeddings)
            loss = criterion(pred, latents)
            loss = ((loss * loss_weight).sum(dim=[1, 2]) / loss_weight.sum(dim=[1, 2])).mean()
            loss_adjusted = loss / args.grad_accum_steps

        acc = (pred.argmax(1) == latents).float()
        acc = acc.mean()
        if not torch.isnan(loss_adjusted):
            if it % args.grad_accum_steps == 0 or it == args.max_iters:
                loss_adjusted.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                with model.no_sync():
                    loss_adjusted.backward()
        else:
            print(f"Encountered NaN loss in iteration {it}.")
            skipped += 1

        pbar.set_postfix({
            'bs': images.size(0),
            'loss': loss_adjusted.item(),
            'acc': acc.item(),
            'grad_norm': grad_norm.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'total_steps': scheduler.last_epoch,
            'skipped': skipped,
        })
        wandb.log({
            'loss': loss_adjusted.item(),
            'acc': acc.item(),
            'grad_norm': grad_norm.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'total_steps': scheduler.last_epoch,
        })

        if it == 1 or it % args.print_every == 0 or it == args.max_iters:
            # if main_node:
            print(f"ITER {it}/{args.max_iters} - loss {loss_adjusted}")

            if it % args.extra_ckpt_every == 0:
                torch.save({
                    'state_dict': model.module.generator.state_dict(),
                    'effnet_state_dict': model.module.effnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_last_step': scheduler.last_epoch,
                    'iter': it,
                    'grad_scaler_state_dict': scaler.state_dict(),
                    'wandb_run_id': run_id,
                }, os.path.join(args.save_checkpoint_path, args.run_name, f"model_stage_B_{it}.pt"))
            torch.save({
                'state_dict': model.module.generator.state_dict(),
                'effnet_state_dict': model.module.effnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_last_step': scheduler.last_epoch,
                'iter': it,
                'grad_scaler_state_dict': scaler.state_dict(),
                'wandb_run_id': run_id,
            }, os.path.join(args.save_checkpoint_path, args.run_name, f"model_stage_B.pt"))

            model.eval()
            images, captions = next(dataloader_iterator)
            while images.size(0) < 8:
                _images, _captions = next(dataloader_iterator)
                images = torch.cat([images, _images], dim=0)
                captions += _captions
            images, captions = images[:8].to(device), captions[:8]
            with torch.no_grad():
                # CLIP stuff
                clip_tokens = clip_tokenizer(captions, truncation=True, padding="max_length",
                                             max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
                clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

                clip_tokens_uncond = clip_tokenizer([""] * len(captions), truncation=True, padding="max_length",
                                                    max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(
                    device)
                clip_embeddings_uncond = clip_model(**clip_tokens_uncond).last_hidden_state
                # ---

                # Efficientnet stuff
                effnet_embeddings = model.module.effnet(effnet_preprocess(images))
                effnet_embeddings_uncond = torch.zeros_like(effnet_embeddings)
                # ---

                t = (1 - torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
                latents = vqmodel.encode(images)[2]
                noised_latents, mask = model.module.generator.add_noise(latents, t)
                pred = model.module.generator(noised_latents, t, effnet_embeddings, clip_text_embeddings)
                pred_tokens = pred.div(0.1).softmax(dim=1).permute(0, 2, 3, 1) @ vqmodel.vquantizer.codebook.weight.data
                pred_tokens = vqmodel.vquantizer.forward(pred_tokens, dim=-1)[-1]
                sampled = sample(model.module.generator, {'effnet': effnet_embeddings, 'byt5': clip_text_embeddings},
                                 (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                 unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                       'byt5': clip_embeddings_uncond})
                sampled_noimg = sample(model.module.generator,
                                       {'effnet': effnet_embeddings, 'byt5': clip_text_embeddings},
                                       (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                       unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                             'byt5': clip_embeddings_uncond})

                noised_images = vqmodel.decode_indices(noised_latents).clamp(0, 1)
                pred_images = vqmodel.decode_indices(pred_tokens).clamp(0, 1)
                sampled_images = vqmodel.decode_indices(sampled).clamp(0, 1)
                sampled_images_noimg = vqmodel.decode_indices(sampled_noimg).clamp(0, 1)
            model.train()

            torchvision.utils.save_image(torch.cat([
                torch.cat([i for i in images.cpu()], dim=-1),
                torch.cat([i for i in noised_images.cpu()], dim=-1),
                torch.cat([i for i in pred_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images_noimg.cpu()], dim=-1),
            ], dim=-2), f'{args.output_path}/{it:06d}.jpg')

            log_data = [[captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(sampled_images_noimg[i])] + [
                wandb.Image(images[i])] for i in range(len(images))]
            log_table = wandb.Table(data=log_data, columns=["Captions", "Sampled", "Sampled noimg", "Orig"])
            wandb.log({"Log": log_table})


if __name__ == '__main__':
    train(0)
