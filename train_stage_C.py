import argparse
import os
import shutil
import time
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
from torchtools.utils import Diffuzz
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from transformers.utils import is_torch_bf16_available, is_torch_tf32_available
from warmup_scheduler import GradualWarmupScheduler
from webdataset.handlers import warn_and_continue

from modules import Paella, sample, EfficientNetEncoder, Prior
from utils import WebdatasetFilter, transforms, effnet_preprocess, identity, ImageTextDataset
from vqgan import VQModel

transformers.utils.logging.set_verbosity_error()

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
    "--stage_b_path",
    type=str,
    default="",
    help="The path to where your stage B model is, trained prior",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default=None,
    help="The wandb project you wish to use",
)
parser.add_argument(
    "--wadnv_entity",
    type=str,
    default=None,
    help="The wandb entity you wish to use",
)


@dataclass
class Arguments:
    updates = 1500000
    warmup_updates = 10000
    ema_start = 5000
    ema_every = 100
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
    stage_b_path = ""

    wandb_project = None
    wandb_entity = None


args = parser.parse_args(namespace=Arguments())


def train(gpu_id):
    device = torch.device(gpu_id)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.save_checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_checkpoint_path, args.run_name), exist_ok=True)

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
        ).shuffle(44, handler=warn_and_continue).decode(
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
    dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=2, pin_memory=False)
    print("REAL BATCH SIZE:", real_batch_size)

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

    diffuzz = Diffuzz(device=device)

    # - CLIP text encoder
    clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_path=args.cache_path).to(
        device).eval().requires_grad_(False)
    clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_path=args.cache_path)
    # - EfficientNet -
    pretrained_checkpoint = torch.load(args.stage_b_path, map_location=device)

    effnet = EfficientNetEncoder(effnet="efficientnet_v2_l").to(device)
    effnet.load_state_dict(pretrained_checkpoint['effnet_state_dict'])
    effnet.eval().requires_grad_(False)

    # - Paella Model as generator -
    generator = Paella(byt5_embd=1024).to(device)
    generator.load_state_dict(pretrained_checkpoint['state_dict'])
    generator.eval().requires_grad_(False)

    del pretrained_checkpoint

    # - Diffusive Imagination Combinatrainer, a.k.a. Risotto - 
    model = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])

    model_ema = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24).to(device).eval().requires_grad_(False)

    # load checkpoints & prepare ddp
    if checkpoint is not None:
        if 'ema_state_dict' in checkpoint:
            model_ema.load_state_dict(checkpoint['ema_state_dict'])
        else:
            model_ema.load_state_dict(model.state_dict())

    # - SETUP WANDB -
    if checkpoint is not None and not args.generate_new_wandb_id:
        try:
            run_id = checkpoint['wandb_run_id']
        except KeyError:
            run_id = wandb.util.generate_id()
    else:
        run_id = wandb.util.generate_id()
    wandb.init(project=args.wandb_project, name=args.run_name, entity=args.wandb_entity, id=run_id, resume="allow")

    print("Num trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # SETUP OPTIMIZER, SCHEDULER & CRITERION
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)  # eps=1e-4
    # optimizer = StableAdamW(model.parameters(), lr=lr) # eps=1e-4
    # optimizer = Lion(model.parameters(), lr=lr / 3) # eps=1e-4
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_updates)
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except KeyError:
            print("Failed loading optimizer, skipping...")

        try:
            scheduler.last_epoch = checkpoint['scheduler_last_step']
        except KeyError:
            print("Failed loading scheduler, skipping...")

        scaler = torch.cuda.amp.GradScaler()
        if checkpoint is not None and 'grad_scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

    start_iter = 1
    grad_norm = torch.tensor(0, device=device)
    if checkpoint is not None:
        try:
            start_iter = checkpoint['scheduler_last_step'] * args.grad_accum_steps + 1
        except KeyError:
            start_iter = 1
        print("RESUMING TRAINING FROM ITER ", start_iter)

    ema_loss = None
    if checkpoint is not None:
        try:
            ema_loss = checkpoint['metrics']['ema_loss']
        except KeyError:
            ema_loss = None

    if checkpoint is not None:
        del checkpoint  # cleanup memory
        torch.cuda.empty_cache()

        # -------------- START TRAINING --------------
    print("Everything prepared, starting training now....")
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, args.max_iters + 1))
    model.train()
    for it in pbar:
        bls = time.time()
        images, captions = next(dataloader_iterator)
        ble = time.time() - bls
        images = images.to(device)

        with torch.no_grad():
            effnet_features = effnet(effnet_preprocess(images))
            with torch.cuda.amp.autocast(dtype=_float16_dtype):
                if np.random.rand() < 0.05:  # 90% of the time, drop the CLIP text embeddings (independently)
                    clip_captions = [''] * len(captions)  # 5% of the time drop all the captions
                else:
                    clip_captions = captions
                clip_tokens = clip_tokenizer(clip_captions, truncation=True, padding="max_length",
                                             max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
                clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

            t = (1 - torch.rand(images.size(0), device=device)).mul(1.08).add(0.001).clamp(0.001, 1.0)
            noised_embeddings, noise = diffuzz.diffuse(effnet_features, t)

        with torch.cuda.amp.autocast(dtype=_float16_dtype):
            pred_noise = model(noised_embeddings, t, clip_text_embeddings)
            loss = nn.functional.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * diffuzz.p2_weight(t)).mean() / args.grad_accum_steps

        if it % args.grad_accum_steps == 0 or it == args.max_iters:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if (it % args.ema_every == 0 or it == args.max_iters):
                if it < args.ema_start:
                    model_ema.load_state_dict(model.state_dict())
                else:
                    model_ema.update_weights_ema(model, beta=args.ema_beta)
        else:
            with model.no_sync():
                loss_adjusted.backward()

        ema_loss = loss.mean().item() if ema_loss is None else ema_loss * 0.99 + loss.mean().item() * 0.01

        pbar.set_postfix({
            'bs': images.size(0),
            'batch_loading': ble,
            'loss': loss.mean().item(),
            'loss_adjusted': loss_adjusted.item(),
            'ema_loss': ema_loss,
            'grad_norm': grad_norm.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'total_steps': scheduler.last_epoch,
        })

        wandb.log({
            'loss': loss.mean().item(),
            'loss_adjusted': loss_adjusted.item(),
            'ema_loss': ema_loss,
            'grad_norm': grad_norm.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'total_steps': scheduler.last_epoch,
        })

        if (it == 1 or it % args.print_every == 0 or it == args.max_iters):
            tqdm.write(f"ITER {it}/{args.max_iters} - loss {ema_loss}")

            if it % args.extra_ckpt_every == 0:
                torch.save({
                    'state_dict': model.state_dict(),
                    'ema_state_dict': model_ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_last_step': scheduler.last_epoch,
                    'iter': it,
                    'metrics': {
                        'ema_loss': ema_loss,
                    },
                    'grad_scaler_state_dict': scaler.state_dict(),
                    'wandb_run_id': run_id,
                }, os.path.join(args.save_checkpoint_path, args.run_name, f"model_stage_C_{it}.pt"))

            torch.save({
                'state_dict': model.state_dict(),
                'ema_state_dict': model_ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_last_step': scheduler.last_epoch,
                'iter': it,
                'metrics': {
                    'ema_loss': ema_loss,
                },
                'grad_scaler_state_dict': scaler.state_dict(),
                'wandb_run_id': run_id,
            }, os.path.join(args.save_checkpoint_path, args.run_name, f"model_stage_C.pt"))

            model.eval()
            images, captions = next(dataloader_iterator)
            images, captions = images.to(device), captions
            images = images[:10]
            captions = captions[:10]
            with torch.no_grad():
                clip_tokens = clip_tokenizer(captions, truncation=True, padding="max_length",
                                             max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
                clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

                clip_tokens_uncond = clip_tokenizer([''] * len(captions), truncation=True, padding="max_length",
                                                    max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(
                    device)
                clip_text_embeddings_uncond = clip_model(**clip_tokens_uncond).last_hidden_state

                t = (1 - torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
                effnet_features = effnet(effnet_preprocess(images))
                effnet_embeddings_uncond = torch.zeros_like(effnet_features)
                noised_embeddings, noise = diffuzz.diffuse(effnet_features, t)

                with torch.cuda.amp.autocast(dtype=_float16_dtype):
                    pred_noise = model(noised_embeddings, t, clip_text_embeddings)
                    pred = diffuzz.undiffuse(noised_embeddings, t, torch.zeros_like(t), pred_noise)
                    sampled = diffuzz.sample(model, {'c': clip_text_embeddings},
                                             unconditional_inputs={"c": clip_text_embeddings_uncond},
                                             shape=effnet_features.shape, cfg=6)[-1]
                    sampled_ema = diffuzz.sample(model_ema, {'c': clip_text_embeddings},
                                                 unconditional_inputs={"c": clip_text_embeddings_uncond},
                                                 shape=effnet_features.shape, cfg=6)[-1]

                    sampled_images = sample(generator, {'effnet': sampled_ema, 'byt5': clip_text_embeddings},
                                            (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                            unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                  'byt5': clip_text_embeddings_uncond})
                    sampled_images_ema = sample(generator, {'effnet': sampled, 'byt5': clip_text_embeddings}, (
                        clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                                unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                      'byt5': clip_text_embeddings_uncond})
                    sampled_images_original = sample(generator,
                                                     {'effnet': effnet_features, 'byt5': clip_text_embeddings}, (
                                                         clip_text_embeddings.size(0), images.size(-2) // 4,
                                                         images.size(-1) // 4),
                                                     unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                           'byt5': clip_text_embeddings_uncond})
                    sampled_pred = sample(generator, {'effnet': pred, 'byt5': clip_text_embeddings},
                                          (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                          unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                'byt5': clip_text_embeddings_uncond})
                    sampled_noised = sample(generator, {'effnet': noised_embeddings, 'byt5': clip_text_embeddings},
                                            (clip_text_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4),
                                            unconditional_inputs={'effnet': effnet_embeddings_uncond,
                                                                  'byt5': clip_text_embeddings_uncond})

                noised_images = vqmodel.decode_indices(sampled_noised).clamp(0, 1)
                pred_images = vqmodel.decode_indices(sampled_pred).clamp(0, 1)
                sampled_images_original = vqmodel.decode_indices(sampled_images_original).clamp(0, 1)
                sampled_images = vqmodel.decode_indices(sampled_images).clamp(0, 1)
                sampled_images_ema = vqmodel.decode_indices(sampled_images_ema).clamp(0, 1)
            model.train()

            torchvision.utils.save_image(torch.cat([
                torch.cat([i for i in images.cpu()], dim=-1),
                torch.cat([i for i in noised_images.cpu()], dim=-1),
                torch.cat([i for i in pred_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images_ema.cpu()], dim=-1),
                torch.cat([i for i in sampled_images_original.cpu()], dim=-1),
            ], dim=-2), f'{args.output_path}/{it:06d}.jpg')

            log_data = [[captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(sampled_images_ema[i])] + [
                wandb.Image(sampled_images_original[i])] + [wandb.Image(images[i])] for i in range(len(images))]
            log_table = wandb.Table(data=log_data,
                                    columns=["Captions", "Sampled", "Sampled EMA", "Sampled Original", "Orig"])
            wandb.log({"Log": log_table})
            del clip_tokens, clip_text_embeddings, clip_tokens_uncond, clip_text_embeddings_uncond, t, effnet_features, effnet_embeddings_uncond
            del noised_embeddings, noise, pred_noise, pred, sampled, sampled_ema, sampled_images, sampled_images_ema, sampled_images_original
            del sampled_pred, sampled_noised, noised_images, pred_images, log_data, log_table


if __name__ == '__main__':
    train(0)
