import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch_model import SizedGenerator, SizedDiscriminator
from model import SimpleDiscriminator, SimpleGenerator
from tqdm import tqdm, trange
import numpy as np
import os

import params as P
from dataloaders import get_dataloader, AVAIL_DATALOADERS
from utils import load_trained_generator, load_trained_disc

def save(path, epoch, model, optimizer, scheduler):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()},
        path)
    return

def load(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def ac_loss(input, disc):
    return torch.mean(torch.abs(input - disc.forward(input)))  # pixelwise L1 - for each pixel for each image in the batch

def main(args):

    # Look for starting checkpoints (if avaialble)
    # TODO - left off here
    # Could have a folder for gen_128_ckpts, gen_256_ckpts, etc
    # inside each folder, we have epoch checkpoints for resuming after crash/kill
    if available_checkpoints_for_resuming:
        gen_checkpoint=...
        disc_checkpoint=...
    else:
        gen_checkpoint=None
        disc_checkpoint=None

    for l in args.latent_dims:
        gen_checkpoint, disc_checkpoint = single_latent_dim(args, l, gen_checkpoint, disc_checkpoint)


def single_latent_dim(args, latent_dim, gen_checkpoint=None, disc_checkpoint=None):
    """
    We warm-start both the gen and disc to keep things "balanced".
    """
    checkpoint_path = f"checkpoints/{args.dataset}_{args.run_name}"
    tensorboard_path = f"tensorboard_logs/{args.dataset}_{args.run_name}"
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tensorboard_path)

    dataloader = get_dataloader(args.dataset_dir, args.n_train, True)

        # ways to deal with linear layer (assume we are doing 64 --> 512)
        # - fix the first 64 in each row, random the rest
        # - F.interpolate(64, 512)
        # - totally random init 512

    # Create or load the Generator
    gen_args = [latent_dim]
    gen_kwargs = {'act': args.output_activ}
    if gen_checkpoint is not None:
        print(f"loading generator checkpoint: {gen_checkpoint}")
        gen = load_trained_generator(SimpleGenerator, gen_checkpoint, device, *gen_args, **gen_kwargs)
        gen.linear = nn.Linear(
                latent_dim, gen.initial_size**2 * gen.ch ,bias=False)
    else:
        gen = SimpleGenerator(*gen_args, **gen_kwargs).to(device)

    # Create or load the Discriminator
    d_args = [latent_dim, P.num_filters, P.size, P.num_ups, args.output_activ]
    d_kwargs = {}
    if disc_checkpoint is not None:
        print(f"loading discriminator checkpoint: {disc_checkpoint}")
        disc = load_trained_generator(SimpleDiscriminator, disc_checkpoint, device, *d_args, **d_kwargs)
        disc.linear = nn.Linear(disc.shape_after_conv, latent_dim, bias=False)
        disc.decoder.linear = nn.Linear(
                latent_dim, disc.decoder.initial_size**2 * disc.decoder.ch, bias=False)

    else:
        disc = SimpleDiscriminator(*d_args, **d_kwargs).to(device)

    gen.train()
    disc.train()

    print(disc)
    print(gen)

    if torch.cuda.device_count() > 1:
        gen = torch.nn.DataParallel(gen)
        disc = torch.nn.DataParallel(disc)

    # NOTE - might consider changing optimizer+scheduler to something like:
    # adamW, lr=0.0002, betas=(0.5, 0.999)
    # CosineAnnealingLR(...T_max=args.epochs)

    gen_optimizer = torch.optim.Adam(gen.parameters(), P.lr)
    gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, gamma=0.95, step_size=P.lr_update_step)
    disc_optimizer = torch.optim.Adam(disc.parameters(), P.lr)
    disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, gamma=0.95, step_size=P.lr_update_step)

    current_checkpoint = 0
    if (not os.path.exists(checkpoint_path)):
        os.makedirs(checkpoint_path)
    else:
        print("Restoring from checkpoint...")
        paths = os.listdir(checkpoint_path)
        try:
            available = sorted(set([int(x.split(".")[1]) for x in paths]))

            # Find a checkpoint that both gen AND disc have reached
            # Reaching zero will cause IndexError during pop()
            while True:
                latest_idx = available.pop()
                latest_disc = os.path.join(checkpoint_path, f"disc_ckpt.{latest_idx}.pt")
                latest_gen = os.path.join(checkpoint_path, f"gen_ckpt.{latest_idx}.pt")
                if os.path.exists(latest_disc) and os.path.exists(latest_gen):
                    break

            current_checkpoint = latest_idx
            disc_epoch = load(latest_disc, disc, disc_optimizer, disc_scheduler)
            gen_epoch = load(latest_gen, gen, gen_optimizer, gen_scheduler)
            assert disc_epoch == gen_epoch, 'Checkpoint contents are mismatched!'
            print(f"Loaded checkpoint {current_checkpoint}")
        except Exception as e:
            print(e)
            print("Unable to load from checkpoint.")

    k = 0

    # Uniform from -1 to 1
    const_sample = ((2 * torch.rand((P.batch_size, latent_dim), dtype=torch.float)) - 1).to(device)

    n_gen_param = sum([x.numel() for x in gen.parameters() if x.requires_grad])
    n_disc_param = sum([x.numel() for x in disc.parameters() if x.requires_grad])
    print(f"{n_gen_param + n_disc_param} Trainable Parameters")

    for e in trange(current_checkpoint, args.epochs, initial=current_checkpoint, leave=True):
        for i, img_batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
            disc_optimizer.zero_grad()
            gen_optimizer.zero_grad()

            # TODO - the paper discusses using this carry weight and decaying it over first epoch
            # Is this important for performance?
            #if e == 0 and P.carry == True:
            #    carry_weight = (1 - (i / len(dataloader)))/2
            #else:
            #    carry_weight = 0

            img_batch = img_batch.to(device)

            # Uniform from -1 to 1
            d_latent_sample = ((2 * torch.rand((P.batch_size, latent_dim), dtype=torch.float)) - 1).to(device)
            g_latent_sample = ((2 * torch.rand((P.batch_size, latent_dim), dtype=torch.float)) - 1).to(device)

            batch_ac_loss = ac_loss(img_batch, disc)
            d_fake_ac_loss = ac_loss(gen.forward(d_latent_sample).detach(), disc)
            g_fake_ac_loss = ac_loss(gen.forward(g_latent_sample), disc)

            def d_loss():
                loss = batch_ac_loss - k * d_fake_ac_loss
                loss.backward()
                return loss

            def g_loss():
                loss = g_fake_ac_loss
                loss.backward()
                return loss

            disc_optimizer.step(d_loss)
            gen_optimizer.step(g_loss)
            disc_scheduler.step()
            gen_scheduler.step()

            k = k + P.prop_gain * (P.gamma * batch_ac_loss.item() - g_fake_ac_loss.item())

            m = ac_loss(img_batch, disc) + torch.abs(P.gamma * batch_ac_loss - g_fake_ac_loss)
            writer.add_scalar("Convergence", m, len(dataloader) * e + i)

            if (i % P.log_every == 0):
                ex_img = gen.forward(g_latent_sample)[0]
                writer.add_image("Generator Output - Random - Raw", ex_img, len(dataloader) * e + i)
                writer.add_image("Generator Output - Random - Clamp", torch.clamp(ex_img, 0, 1), len(dataloader) * e + i)
                ex_img -= ex_img.min()
                ex_img /= ex_img.max()
                writer.add_image("Generator Output - Random - Normalize", ex_img, len(dataloader) * e + i)
                ex_img_const = gen.forward(const_sample)[0]
                writer.add_image("Generator Output - Constant - Raw", ex_img_const, len(dataloader) * e + i)
                writer.add_image("Generator Output - Constant - Clamp", torch.clamp(ex_img_const, 0, 1), len(dataloader) * e + i)
                ex_img_const -= ex_img_const.min()
                ex_img_const /= ex_img_const.max()
                writer.add_image("Generator Output - Constant - Normalize", ex_img_const, len(dataloader) * e + i)

        save(os.path.join(checkpoint_path, f"gen_ckpt.{e}.pt"), e, gen, gen_optimizer, gen_scheduler)
        save(os.path.join(checkpoint_path, f"disc_ckpt.{e}.pt"), e, disc, disc_optimizer, disc_scheduler)


def str2list(s):
    return [int(x) for x in s.split(',')]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=AVAIL_DATALOADERS, required=True)
    p.add_argument('--latent_dims', type=str2list, default=[64], help='Comma-separated list of dimensions, no spaces. For example, --latent_dims 64,128,256 ')
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--run_name', required=True)
    p.add_argument('--n_train', type=int, default=-1)
    p.add_argument('--output_activ', choices=['elu','sigmoid'], default='elu')
    args = p.parse_args()
    main(args)
