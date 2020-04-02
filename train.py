import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_model import SizedGenerator, SizedDiscriminator
from model import SimpleDiscriminator, SimpleGenerator
from tqdm import tqdm, trange
import os

import params as P
from dataloaders import get_dataloader, AVAIL_DATALOADERS

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

def main(dataset, dataset_path, run_name, n_train, output_activ, epochs, latent_dim):
    checkpoint_path = f"checkpoints/{dataset}_{run_name}"
    tensorboard_path = f"tensorboard_logs/{dataset}_{run_name}"
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tensorboard_path)

    dataloader = get_dataloader(dataset_path, n_train, True)
    # TODO - useful print? print(f"FolderDataset: {dataloader.dataset}")

#    gen = SizedGenerator(latent_dim, P.num_filters, P.size, P.num_ups, output_activ).to(device)
    gen = SimpleGenerator(latent_dim, act=output_activ).to(device)
#    disc = SizedDiscriminator(latent_dim, P.num_filters, P.size, P.num_ups, output_activ).to(device)
    disc = SimpleDiscriminator(latent_dim, P.num_filters, P.size, P.num_ups, output_activ).to(device)
    print(disc)
    print(gen)

    if torch.cuda.device_count() > 1:
        gen = torch.nn.DataParallel(gen)
        disc = torch.nn.DataParallel(disc)

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

    for e in trange(current_checkpoint, epochs, initial=current_checkpoint, leave=True):
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=AVAIL_DATALOADERS, required=True)
    p.add_argument('--latent_dim', type=int, default=64)
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--run_name', required=True)
    p.add_argument('--n_train', type=int, default=-1)
    p.add_argument('--output_activ', choices=['elu','sigmoid'], default='elu')
    args = p.parse_args()
    main(args.dataset, args.dataset_dir, args.run_name, args.n_train, args.output_activ, args.epochs, args.latent_dim)
