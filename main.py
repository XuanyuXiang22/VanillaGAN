import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as trans
from network import Generartor, Discriminator

# Things to try:
# 1. What happens if you use larger network?
# 2. Better normalization with BatchNorm
# 3．Different learning rate (is there a better one) ?
# 4. Change architecture to a CNN


if __name__ == "__main__":
    # Hyperparameters, GAN is greatly sensitive to hyperparameters!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64
    img_dim = 28 * 28 * 1
    batch_size = 32
    num_epochs = 50

    # models
    disc = Discriminator(img_dim).to(device)
    gen = Generartor(z_dim, img_dim).to(device)

    # datasets
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    transform = trans.Compose([
        trans.ToTensor(), trans.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="/home/ubuntu/dev/sda1/data/data_xuanyu", transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optimizers
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

    # criterion
    criterion = torch.nn.BCELoss()

    # 结果可视化
    write_fake = SummaryWriter(r"runs/fake")
    write_real = SummaryWriter(r"runs/real")
    step = 0

    # training
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            # Train discriminator: max log(D(real)) + log(1 - D(G(z)))
            noise = torch.randn((batch_size, z_dim)).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)  # flatten
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).view(-1)  # detach 保留G(z)的计算图，使计算生成器损失时可以复用计算图
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            # Train generator: min log(1 - D(G(z))) saturating gradients --> max log(D(G(z)))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            # 训练可视化
            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(loader)}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    write_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
                    write_real.add_image("MNIST Real Images", img_grid_real, global_step=step)\

                    step += 1