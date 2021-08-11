import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils import *

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64


def train_gan():

    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")

    dataloader = prepare_dataset(IMAGE_SIZE, CHANNELS_IMG, BATCH_SIZE)

    generator, discriminator = initialize_models(
        NOISE_DIM, CHANNELS_IMG, FEATURES_GEN, FEATURES_DISC, device
    )

    optim_generator = optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    optim_discriminator = optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    criterion = nn.BCELoss()

    generator.train()
    discriminator.train()
    step = 0

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = generator(noise)

            discriminator_real = discriminator(real).reshape(-1)
            discriminator_loss_real = criterion(
                discriminator_real, torch.ones_like(discriminator_real)
            )
            discriminator_fake = discriminator(fake.detach()).reshape(-1)
            discriminator_loss_fake = criterion(
                discriminator_fake, torch.zeros_like(discriminator_fake)
            )
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            discriminator.zero_grad()
            discriminator_loss.backward()
            optim_discriminator.step()

            output = discriminator(fake).reshape(-1)
            generator_loss = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            generator_loss.backward()
            optim_generator.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {discriminator_loss:.4f}, loss G: {generator_loss:.4f}"
                )

                with torch.no_grad():
                    fake = generator(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1


if __name__ == "__main__":
    train_gan()
