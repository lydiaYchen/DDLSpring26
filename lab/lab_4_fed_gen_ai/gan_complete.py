import datasets
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from commons.utils import get_device
from matplotlib.figure import Figure
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import trange

DEVICE = get_device()
torch.set_float32_matmul_precision("high")
IMAGE_SIZE = 128
preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}
TRAIN_DATA = datasets.load_dataset(
    "huggan/smithsonian_butterflies_subset",
    split="train").select_columns(["image"])
TRAIN_DATA.set_transform(transform)
nr_clients = 4
CLIENT_SPLIT = [
    TRAIN_DATA.shard(num_shards=nr_clients, index=i)
    for i in range(nr_clients)
]


def plot(images: torch.Tensor | list[torch.Tensor], title="") -> Figure:
    grid = make_grid(images, padding=2, normalize=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    ax.axis("off")

    if title != "":
        ax.set_title(title)

    return fig


# https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=128, nc=3) -> None:
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=128) -> None:
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GanFedAvgClient:
    def __init__(
            self, client_data: datasets.Dataset,
            lr: float, batch_size: int, nr_epochs: int,
            beta1: float, seed: int) -> None:
        torch.manual_seed(seed)
        self.gen = Generator().to(DEVICE)
        self.discr = Discriminator().to(DEVICE)
        self.gen.compile()
        self.discr.compile()
        self.loader_train = DataLoader(
            client_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.lr = lr
        self.nr_epochs = nr_epochs
        self.beta1 = beta1
        self.opt_gen: Adam
        self.opt_discr: Adam
        self.criterion = F.binary_cross_entropy_with_logits
        self.round_loss_gen: float
        self.round_loss_discr: float

    def train_epoch(self) -> None:
        for i, data in enumerate(self.loader_train):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            self.discr.zero_grad()
            # format batch
            real_cpu = data["images"].to(DEVICE)
            b_size = real_cpu.size(0)
            # apply label smoothing for better learning
            label = torch.empty(b_size, device=DEVICE).uniform_(0.9, 1.0)
            # forward pass real batch through D
            output = self.discr(real_cpu).view(-1)
            # calculate loss on all-real batch
            err_discr_real = self.criterion(output, label)
            # calculate gradients for D in backward pass
            err_discr_real.backward()
            # D_x = output.mean().item()

            ## Train with all-fake batch
            # generate batch of latent vectors
            noise = torch.randn(b_size, self.gen.nz, 1, 1, device=DEVICE)
            # generate fake image batch with G
            fake = self.gen(noise)
            label.fill_(0.)
            # classify all fake batch with D
            output = self.discr(fake.detach()).view(-1)
            # calculate D's loss on the all-fake batch
            err_discr_fake = self.criterion(output, label)
            # calculate the gradients for this batch, accumulated (summed) with previous gradients
            err_discr_fake.backward()
            # update D based on accumulated error over the fake and the real batches
            self.opt_discr.step()
            self.round_loss_gen += (err_discr_real + err_discr_fake).item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.gen.zero_grad()
            label.fill_(1.)  # fake labels are real for generator cost
            # since we just updated D, perform another forward pass of all-fake batch through D
            output = self.discr(fake).view(-1)
            # calculate G's loss based on this output
            err_gen = self.criterion(output, label)
            # calculate gradients for G
            err_gen.backward()
            # update G
            self.opt_gen.step()
            self.round_loss_discr += err_gen.item()

    def update(
            self, gen_state_dict: dict[str, torch.Tensor],
            discr_state_dict: dict[str, torch.Tensor],
            seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # use `state_dict` instead of `parameters`` to include batch norm buffers
        self.gen.load_state_dict(gen_state_dict)
        self.discr.load_state_dict(discr_state_dict)
        torch.manual_seed(seed)
        self.round_loss_gen = 0.
        self.round_loss_discr = 0.
        self.gen.train()
        self.discr.train()
        self.opt_gen = Adam(self.gen.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.opt_discr = Adam(self.discr.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        for _epoch in range(self.nr_epochs):
            self.train_epoch()

        self.round_loss_gen /= self.nr_epochs * len(self.loader_train)
        self.round_loss_discr /= self.nr_epochs * len(self.loader_train)

        return (self.gen.state_dict(), self.discr.state_dict())


class GanFedAvgServer:
    def __init__(
            self, lr: float, batch_size: int, client_subsets: list[datasets.Dataset],
            client_fraction: float, nr_local_epochs: int,
            beta1: float, seed: int) -> None:
        self.seed = seed
        torch.manual_seed(seed)
        self.gen = Generator().to(DEVICE).apply(weights_init)
        self.discr = Discriminator().to(DEVICE).apply(weights_init)

        self.nr_clients = len(client_subsets)
        self.client_fraction = client_fraction
        self.client_sample_counts = [len(subset) for subset in client_subsets]
        self.nr_clients_per_round = max(
            1, round(client_fraction * self.nr_clients))
        self.rng = npr.default_rng(seed)

        self.clients = [
            GanFedAvgClient(subset, lr, batch_size, nr_local_epochs, beta1, seed)
            for subset in client_subsets
        ]

    def run(self, nr_rounds: int) -> None:
        for nr_round in (pbar := trange(nr_rounds, desc="Rounds", leave=True)):
            gen_state_dict = self.gen.state_dict()
            discr_state_dict = self.discr.state_dict()
            indices_chosen_clients = self.rng.choice(
                self.nr_clients, self.nr_clients_per_round, replace=False)
            chosen_sum_nr_samples = sum(
                self.client_sample_counts[i] for i in indices_chosen_clients)
            gen_aggr_dict = {
                k: torch.zeros_like(v) for k, v in gen_state_dict.items()}
            discr_aggr_dict = {
                k: torch.zeros_like(v) for k, v in discr_state_dict.items()}

            for c_i in indices_chosen_clients:
                ind = int(c_i)
                client_round_seed = self.seed + ind + 1 + nr_round * self.nr_clients_per_round
                client_gen_dict, client_discr_dict = self.clients[ind].update(
                    gen_state_dict, discr_state_dict, client_round_seed)
                frac = self.client_sample_counts[ind] / chosen_sum_nr_samples

                for k, v in client_gen_dict.items():
                    # BatchNorm2d.num_batches_tracked has dtype long, not float
                    # thus, the explicit cast helps avoid errors
                    gen_aggr_dict[k] += (v * frac).to(v.dtype)

                for k, v in client_discr_dict.items():
                    discr_aggr_dict[k] += (v * frac).to(v.dtype)

            self.gen.load_state_dict(gen_aggr_dict)
            self.discr.load_state_dict(discr_aggr_dict)

            loss_sum_gen = sum(
                self.clients[c_i].round_loss_gen for c_i in indices_chosen_clients)
            loss_sum_discr = sum(
                self.clients[c_i].round_loss_discr for c_i in indices_chosen_clients)
            pbar.set_postfix(loss_sum_gen=loss_sum_gen, loss_sum_discr=loss_sum_discr)

    def sample(self, nr_samples: int, sample_seed=0) -> torch.Tensor:
        print(f"Generating {nr_samples} images...")
        self.gen.eval()
        torch.manual_seed(sample_seed)

        with torch.no_grad():
            fake = self.gen(torch.randn(nr_samples, self.gen.nz, 1, 1, device=DEVICE))
            fake = fake.detach().cpu()

        return fake
