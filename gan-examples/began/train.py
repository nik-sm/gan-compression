import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch_model import SizedGenerator, SizedDiscriminator
from tqdm import tqdm
import numpy as np
import os

batch_size = 16    # --> Size of training batch
size = 128  # --> Size of images for training
num_ups = 4 # --> Number of upsampling layers
test_folder = "data"  # --> Location of image folder
train_epochs = 30  # --> Number of training epochs
log_every = 100  # --> Number of iterations after which to log

latent_dim = 64  # --> Dimensionality of latent codes
ngpu = 1  # --> Number of GPUs
num_filters = 128  # --> Number of filters to use in each layer

lr = 0.0001  # --> Learning rate for Adam optimizers
lr_update_step = 12000  # --> Half the learning rate after this many steps
lr_min = 0.00002 # --> Minimum learning rate

gamma = 0.5  # --> Image diversity hyperparameter: [0,1]
prop_gain = 0.001  # --> Proportional gain for k
cuda = True  # --> Use cuda
multi_gpu = True # --> Use multiple GPUs

carry = False # --> Use disappearing residuals

trans = transforms.Compose([transforms.CenterCrop(130), transforms.Resize((size, size)), transforms.ToTensor()])
dataset = datasets.ImageFolder(test_folder, transform=trans)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)

gen = SizedGenerator(latent_dim, num_filters, size, num_ups, ngpu)
disc = SizedDiscriminator(latent_dim, num_filters, size, num_ups, ngpu)

#gen = gen.half()
#disc = disc.half()

current_checkpoint = 0
if (not os.path.exists("checkpoints")):
    os.mkdir("checkpoints")
else:
    print("Restoring from checkpoint...")
    paths = os.listdir("checkpoints")
    try:
        latest_idx = sorted([int(x.split(".")[0].split("_")[2]) for x in paths], reverse=True)[0]
        latest_disc = "disc_ckpt_{0}.pt".format(latest_idx)
        latest_gen = "gen_ckpt_{0}.pt".format(latest_idx)
        current_checkpoint = latest_idx
        disc.load_state_dict(torch.load(os.path.join("checkpoints", latest_disc)))
        gen.load_state_dict(torch.load(os.path.join("checkpoints", latest_gen)))
        print("Loaded checkpoint {0}".format(current_checkpoint))
    except:
        print("Unable to load from checkpoint. ")

writer = SummaryWriter()

gen_optimizer = torch.optim.Adam(gen.parameters(), lr)
disc_optimizer = torch.optim.Adam(disc.parameters(), lr)

if (cuda):
    gen.cuda()
    disc.cuda()

if (multi_gpu):
    gen = torch.nn.DataParallel(gen)
    disc = torch.nn.DataParallel(disc)

k = 0


def ac_loss(input):
    return torch.mean(torch.abs(input - disc.forward(input)))  # pixelwise L1 - for each pixel for each image in the batch

const_sample = torch.tensor(np.random.uniform(low=-1, high=1, size=(1, latent_dim)), dtype=torch.float)

print("One epoch is {0} batches".format(len(dataloader)))
print("{0} Trainable Parameters".format(sum([x.numel() for x in gen.parameters() if x.requires_grad]) + sum([x.numel() for x in disc.parameters() if x.requires_grad])))

for e in range(train_epochs - current_checkpoint):
    print(f"Epoch {e}")
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if(e == 0 and carry == True):
            carry_weight = (1 - (i / len(dataloader)))/2
        else:
            carry_weight = 0

        img_batch = data[0]

        if (cuda):
            img_batch = img_batch.cuda()

        d_latent_sample = np.random.uniform(low=-1, high=1, size=(batch_size, latent_dim))
        g_latent_sample = np.random.uniform(low=-1, high=1, size=(batch_size, latent_dim))

        d_latent_sample = torch.tensor(d_latent_sample, dtype=torch.float)
        g_latent_sample = torch.tensor(g_latent_sample, dtype=torch.float)

        if (cuda):
            img_batch = img_batch.cuda()
            const_sample = const_sample.cuda()
            d_latent_sample = d_latent_sample.cuda()
            g_latent_sample = g_latent_sample.cuda()

        batch_ac_loss = ac_loss(img_batch)
        d_fake_ac_loss = ac_loss(gen.forward(d_latent_sample).detach())
        g_fake_ac_loss = ac_loss(gen.forward(g_latent_sample))


        def d_loss():
            disc_optimizer.zero_grad()
            loss = batch_ac_loss - k * d_fake_ac_loss
            loss.backward()
            return loss


        def g_loss():
            gen_optimizer.zero_grad()
            loss = g_fake_ac_loss
            loss.backward()
            return loss


        disc_optimizer.step(d_loss)
        gen_optimizer.step(g_loss)

        k = k + prop_gain * (gamma * batch_ac_loss.item() - g_fake_ac_loss.item())

        m = ac_loss(img_batch) + torch.abs(gamma * batch_ac_loss - g_fake_ac_loss)
        writer.add_scalar("Convergence", m, len(dataloader) * e + i)

        if (i % log_every == 0):
            ex_img = gen.forward(g_latent_sample)[0]
            ex_img_const = gen.forward(const_sample)[0]
            writer.add_image("Generator Output - Constant", ex_img_const, len(dataloader) * e + i)
            writer.add_image("Generator Output - Random", ex_img, len(dataloader) * e + i)
        if((i + len(dataloader) * e) % lr_update_step == (lr_update_step-1)):
            print("\nHalving the learning rate...")
            lr = max(lr * 0.95, lr_min)
            for param_group in disc_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in gen_optimizer.param_groups:
                param_group['lr'] = lr

    torch.save(gen.state_dict(), "checkpoints/gen_ckpt_{0}.pt".format(e))
    torch.save(disc.state_dict(), "checkpoints/disc_ckpt_{0}.pt".format(e))

