import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Discriminator, Generator, initialize_weights


def prepare_dataset(image_size, channels_img, batch_size):
    """ Prepare dataset through DataLoader """
    dataset = datasets.MNIST(
        root="dataset/",
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(channels_img)],
                    [0.5 for _ in range(channels_img)],
                ),
            ]
        ),
        download=True,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def initialize_models(noise_dim, channels_img, features_gen, features_disc, device):
    generator = Generator(noise_dim, channels_img, features_gen).to(device)
    discriminator = Discriminator(channels_img, features_disc).to(device)
    initialize_weights(generator)
    initialize_weights(discriminator)

    return generator, discriminator


def print_training_progress(epoch, batch, generator_loss, discriminator_loss):
    """ Print training progress. """
    print(
        "Losses at epoch %5d, after mini-batch %5d: generator %e, discriminator %e"
        % (epoch, batch, generator_loss, discriminator_loss)
    )

