import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.autograd as autograd

from scipy.stats import wasserstein_distance

device = torch.device("mps")


class Dataset(object):
    def __init__(self, train_data, test_data, x_dim, y_dim, traj_len):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.traj_len = traj_len
        self.load_train_data(train_data)
        self.load_test_data(test_data)

    def add_grid_data(self, grid_fn):
        self.grid_test_fn = grid_fn

    def load_train_data(self, data):
        X = data["X"][:, : self.traj_len, :]
        Y = data["Y_s0"]

        self.HMAX = np.max(np.max(X, axis=0), axis=0)

        self.HMIN = np.min(np.min(X, axis=0), axis=0)

        self.X_train = -1 + 2 * (X - self.HMIN) / (self.HMAX - self.HMIN)
        Y = -1 + 2 * (Y - self.HMIN) / (self.HMAX - self.HMIN)
        self.Y_train = np.expand_dims(Y, axis=2)
        self.n_points_dataset = self.X_train.shape[0]

        Xt = np.empty((self.n_points_dataset, self.x_dim, self.traj_len))
        Yt = np.empty((self.n_points_dataset, self.y_dim, 1))
        for j in range(self.n_points_dataset):
            Xt[j] = self.X_train[j].T
            Yt[j] = self.Y_train[j]
        self.X_train_transp = Xt
        self.Y_train_transp = Yt

    def load_test_data(self, data):
        X = data["X"][:, :, : self.traj_len, :]
        Y = data["Y_s0"]

        self.X_test = -1 + 2 * (X - self.HMIN) / (self.HMAX - self.HMIN)

        Y = -1 + 2 * (Y - self.HMIN) / (self.HMAX - self.HMIN)

        self.n_points_test = self.X_test.shape[0]
        self.n_traj_per_point = self.X_test.shape[1]

        Xt = np.empty(
            (self.n_points_test, self.n_traj_per_point, self.x_dim, self.traj_len)
        )

        for j in range(self.n_points_test):
            for k in range(self.n_traj_per_point):
                Xt[j, k] = self.X_test[j, k].T

        self.X_test_transp = Xt
        self.Y_test_transp = np.expand_dims(Y, axis=2)

    def prepare_init_conditions_for_inference(self, Y):
        """
        Expects the Y_s0 portion of the dataset dict.
        """
        Y = -1 + 2 * (Y - self.HMIN) / (self.HMAX - self.HMIN)
        return np.expand_dims(Y, axis=2)

    def rescale_values(self, data):
        """
        Rescale the values back to the original range.
        """
        data = self.HMIN + (data + 1) * (self.HMAX - self.HMIN) / 2
        return data

    def swap_last_two_dimensions(self, data):
        return np.transpose(data, (0, 1, 3, 2))

    def stack_simulations(self, data):
        """
        Example: (100, 50, 31, 7) -> (5000, 31, 7)
        """
        return data.reshape(-1, data.shape[2], data.shape[3])

    def load_grid_test_data(self):
        file = open(self.grid_test_fn, "rb")
        data = pickle.load(file)
        file.close()

        X = data["X"][:, :, : self.traj_len, :]
        Y = data["Y_s0"]

        self.X_test_grid = -1 + 2 * (X - self.HMIN) / (self.HMAX - self.HMIN)

        Y = -1 + 2 * (Y - self.HMIN) / (self.HMAX - self.HMIN)

        self.n_points_grid = self.X_test_grid.shape[0]
        self.n_traj_per_point_grid = self.X_test_grid.shape[1]

        Xt = np.empty(
            (self.n_points_grid, self.n_traj_per_point_grid, self.x_dim, self.traj_len)
        )

        for j in range(self.n_points_grid):
            for k in range(self.n_traj_per_point_grid):
                Xt[j, k] = self.X_test_grid[j, k].T

        self.X_test_grid_transp = Xt
        self.Y_test_grid_transp = np.expand_dims(Y, axis=2)

    def generate_mini_batches(self, n_samples):
        ix = np.random.randint(0, self.X_train_transp.shape[0], n_samples)
        Xb = self.X_train_transp[ix]
        Yb = self.Y_train_transp[ix]

        return Xb, Yb


class Generator(nn.Module):
    def __init__(self, traj_len, latent_dim, x_dim):
        super(Generator, self).__init__()

        self.x_dim = x_dim

        self.init_size = traj_len // int(traj_len / 2)

        self.padd = 1
        self.n_filters = 2 * self.padd + 1
        self.Q = 2
        self.Nch = 512

        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.Nch * self.Q))

        if traj_len == 16:
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose1d(
                    self.Nch + x_dim, 128, 4, stride=2, padding=self.padd
                ),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(128, x_dim, self.n_filters, stride=1, padding=self.padd),
                nn.Tanh(),
            )
        elif traj_len == 32:
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose1d(
                    self.Nch + x_dim, 128, 4, stride=2, padding=self.padd
                ),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(256, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(128, x_dim, self.n_filters, stride=1, padding=self.padd),
                nn.Tanh(),
            )
        elif traj_len == 64:
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose1d(
                    self.Nch + x_dim, 128, 4, stride=2, padding=self.padd
                ),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(256, 512, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(512, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(512, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(128, x_dim, self.n_filters, stride=1, padding=self.padd),
                nn.Tanh(),
            )
        elif traj_len == 128:
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose1d(
                    self.Nch + x_dim, 128, 4, stride=2, padding=self.padd
                ),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(256, 512, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(512, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(512, 1024, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(1024, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(1024, 512, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(512, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(512, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(128, x_dim, self.n_filters, stride=1, padding=self.padd),
                nn.Tanh(),
            )
        else:
            raise Exception(
                "Only trajectory lengths of 16, 32, 64, or 128 are supported."
            )

    def forward(self, noise, conditions):
        conds_flat = conditions.view(conditions.shape[0], -1)
        conds_rep = conds_flat.repeat(1, self.Q).view(
            conditions.shape[0], self.x_dim, self.Q
        )
        noise_out = self.l1(noise)
        noise_out = noise_out.view(noise_out.shape[0], self.Nch, self.Q)
        gen_input = torch.cat((conds_rep, noise_out), 1)
        traj = self.conv_blocks(gen_input)

        return traj


class Discriminator(nn.Module):
    def __init__(self, traj_len, x_dim):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, L):
            padd = 1
            n_filters = 2 * padd + 2
            block = [
                nn.Conv1d(in_filters, out_filters, n_filters, stride=2, padding=padd),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2),
            ]
            block.append(nn.LayerNorm([out_filters, L]))

            return block

        self.model = nn.Sequential(
            *discriminator_block(x_dim, 64, traj_len // 2),
            *discriminator_block(64, 64, traj_len // 4),
        )

        # The height and width of downsampled image
        ds_size = (traj_len + 1) // (2**2)
        self.adv_layer = nn.Sequential(nn.Linear(64 * ds_size, 1))

    def forward(self, trajs, conditions):
        d_in = torch.cat((conditions, trajs), 2)
        out = self.model(d_in)
        out_flat = out.view(out.shape[0], -1)
        validity = self.adv_layer(out_flat)
        return validity


class GanManager:
    def __init__(
        self,
        model_name,
        n_epochs,
        batch_size,
        species_names,
        x_dim,
        traj_len,
        end_time,
        id=None,
    ):
        self.model_name = model_name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.species_names = species_names
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.traj_len = traj_len
        self.end_time = end_time
        self.cuda = True if torch.cuda.is_available() else False

        self.id = id if id else str(np.random.randint(0, 100000))

        self.plots_path = f"trained_gan/{self.model_name}_{self.id}"
        os.makedirs(self.plots_path, exist_ok=True)

        self.MODEL_PATH = f"{self.plots_path}/generator_{self.n_epochs}epochs.pt"
        self.lambda_gp = 10
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.latent_dim = 480
        self.colors = ["blue", "orange"]
        self.leg = ["real", "gen"]

        print(
            f">> Initialised GanManager with model: {self.model_name}, ID: {self.id}, epochs: {self.n_epochs}."
        )

    def init_dataset(self, train_data: dict, test_data: dict):
        self.ds = Dataset(train_data, test_data, self.x_dim, self.y_dim, self.traj_len)
        print(">> Loaded data.")
        print(
            f">> Train shape: X {self.ds.X_train_transp.shape}, Y {self.ds.Y_train_transp.shape}."
        )
        print(
            f">> Test shape: X {self.ds.X_test_transp.shape}, Y {self.ds.Y_test_transp.shape}."
        )

    def init_models(self):
        self.generator = Generator(self.traj_len, self.latent_dim, self.x_dim)
        self.discriminator = Discriminator(self.traj_len, self.x_dim)
        print(f">> Initialised Generator and Discriminator.")

    def compute_gradient_penalty(self, D, real_samples, fake_samples, lab):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples

        alpha = self.Tensor(np.random.random((real_samples.size(0), 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates = D(interpolates, lab)
        fake = Variable(
            self.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False
        )

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.shape[0], self.traj_len * self.x_dim)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def generate_random_conditions(self):
        return (np.random.rand(self.batch_size, self.y_dim, 1) - 0.5) * 2

    def train(self, lr=0.0001, b1=0.5, b2=0.9, n_critic=5):
        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        # Optimizers
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )

        batches_done = 0
        G_losses = []
        D_losses = []
        real_comp = []
        gen_comp = []
        gp_comp = []

        full_G_loss = []
        full_D_loss = []
        for epoch in range(self.n_epochs):
            bat_per_epo = int(self.ds.n_points_dataset / self.batch_size)
            n_steps = bat_per_epo * self.n_epochs

            tmp_G_loss = []
            tmp_D_loss = []

            for i in range(bat_per_epo):
                trajs_np, conds_np = self.ds.generate_mini_batches(self.batch_size)
                # Configure input
                real_trajs = Variable(self.Tensor(trajs_np))
                conds = Variable(self.Tensor(conds_np))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(
                    self.Tensor(
                        np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                    )
                )

                # Generate a batch of images
                fake_trajs = self.generator(z, conds)
                # Real images
                real_validity = self.discriminator(real_trajs, conds)
                # Fake images
                fake_validity = self.discriminator(fake_trajs, conds)
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(
                    self.discriminator, real_trajs.data, fake_trajs.data, conds.data
                )
                # Adversarial loss
                d_loss = (
                    -torch.mean(real_validity)
                    + torch.mean(fake_validity)
                    + self.lambda_gp * gradient_penalty
                )
                real_comp.append(torch.mean(real_validity).item())
                gen_comp.append(torch.mean(fake_validity).item())
                gp_comp.append(self.lambda_gp * gradient_penalty.item())
                tmp_D_loss.append(d_loss.item())
                full_D_loss.append(d_loss.item())

                d_loss.backward(retain_graph=True)
                optimizer_D.step()

                # Train the generator every n_critic steps
                if i % n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_G.zero_grad()
                    gen_conds = Variable(self.Tensor(self.generate_random_conditions()))

                    # Generate a batch of images
                    gen_trajs = self.generator(z, gen_conds)

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(gen_trajs, gen_conds)
                    g_loss = -torch.mean(fake_validity)
                    tmp_G_loss.append(g_loss.item())
                    full_G_loss.append(g_loss.item())
                    g_loss.backward(retain_graph=True)
                    optimizer_G.step()

                    print(
                        f"[Epoch {epoch + 1}/{self.n_epochs}] [Batch {i}/{bat_per_epo}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                    )

                    batches_done += n_critic
            if (epoch + 1) % 500 == 0:
                torch.save(
                    self.generator, self.plots_path + f"/generator_{epoch}epochs.pt"
                )
            D_losses.append(np.mean(tmp_D_loss))
            G_losses.append(np.mean(tmp_G_loss))

        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        axs[0].plot(np.arange(self.n_epochs), G_losses)
        axs[1].plot(np.arange(self.n_epochs), D_losses)
        axs[0].set_title("generator loss")
        axs[1].set_title("critic loss")
        plt.tight_layout()
        fig.savefig(self.plots_path + "/losses.png")
        plt.close()

        fig1, axs1 = plt.subplots(2, 1, figsize=(12, 6))
        axs1[0].plot(np.arange(len(full_G_loss)), full_G_loss)
        axs1[1].plot(np.arange(len(full_D_loss)), full_D_loss)
        axs1[0].set_title("generator loss")
        axs1[1].set_title("critic loss")
        plt.tight_layout()
        fig1.savefig(self.plots_path + "/full_losses.png")
        plt.close()

        fig2, axs2 = plt.subplots(3, 1, figsize=(12, 9))
        axs2[0].plot(np.arange(n_steps), real_comp)
        axs2[1].plot(np.arange(n_steps), gen_comp)
        axs2[2].plot(np.arange(n_steps), gp_comp)
        axs2[0].set_title("real term")
        axs2[1].set_title("generated term")
        axs2[2].set_title("gradient penalty term")
        plt.tight_layout()
        fig2.savefig(self.plots_path + "/components.png")
        plt.close()

        # save the ultimate trained generator
        torch.save(self.generator, self.MODEL_PATH)

    def load_trained_model(self):
        print(">> Loading model from MODEL_PATH: ", self.MODEL_PATH)
        self.generator = torch.load(self.MODEL_PATH)
        self.generator.eval()
        if self.cuda:
            self.generator.cuda()

    def add_time(self, data, endtime):
        a, b, c = data.shape
        timestamps = np.linspace(0, endtime, b).reshape(1, b, 1)
        timestamps = np.repeat(timestamps, a, axis=0)
        return np.concatenate((timestamps, data), axis=2)

    def generate_trajectories_for_init_conditions(
        self, init_conditions, n_sims_per_init_condition
    ):
        init_conditions = self.ds.prepare_init_conditions_for_inference(init_conditions)
        gen_trajectories = np.empty(
            shape=(
                len(init_conditions),
                n_sims_per_init_condition,
                self.x_dim,
                self.traj_len + 1,
            )
        )
        for i, init_condition in enumerate(init_conditions):
            print(f">> Processing initial condition {i + 1} / {len(init_conditions)}.")
            for j in range(n_sims_per_init_condition):
                z_noise = np.random.normal(0, 1, (1, self.latent_dim))
                temp_out = self.generator(
                    Variable(self.Tensor(z_noise)),
                    Variable(self.Tensor([init_condition])),
                )
                temp_out = temp_out.detach().cpu().numpy()[0]
                temp_out = np.hstack((init_condition, temp_out))
                gen_trajectories[i, j] = temp_out

        gen_trajectories = self.ds.swap_last_two_dimensions(gen_trajectories)
        gen_trajectories = self.ds.stack_simulations(gen_trajectories)
        gen_trajectories = self.ds.rescale_values(gen_trajectories)
        gen_trajectories = np.round(gen_trajectories)
        gen_trajectories = self.add_time(gen_trajectories, self.end_time)

        return gen_trajectories

    def compute_test_trajectories(self):
        print(">> Computing test trajectories.")
        n_gen_trajs = self.ds.n_traj_per_point
        self.gen_trajectories = np.empty(
            shape=(self.ds.n_points_test, n_gen_trajs, self.x_dim, self.traj_len)
        )
        for iii in range(self.ds.n_points_test):
            print("Test point nb ", iii + 1, " / ", self.ds.n_points_test)
            for jjj in range(n_gen_trajs):
                z_noise = np.random.normal(0, 1, (1, self.latent_dim))
                temp_out = self.generator(
                    Variable(self.Tensor(z_noise)),
                    Variable(self.Tensor([self.ds.Y_test_transp[iii]])),
                )
                self.gen_trajectories[iii, jjj] = temp_out.detach().cpu().numpy()[0]

        trajs_dict = {"gen_trajectories": self.gen_trajectories}
        file = open(self.plots_path + "/generated_validation_trajectories.pickle", "wb")
        # dump information to that file
        pickle.dump(trajs_dict, file)
        # close the file
        file.close()

    def load_test_trajectories(self):
        file = open(self.plots_path + "/generated_validation_trajectories.pickle", "rb")
        trajs_dict = pickle.load(file)
        file.close()
        self.gen_trajectories = trajs_dict["gen_trajectories"]

    def plot_trajectories(self):
        # PLOT TRAJECTORIES
        n_trajs_to_plot = 10
        print("Plotting test trajectories...")
        tspan = range(self.traj_len)
        for kkk in range(self.ds.n_points_test):
            print("Test point nb ", kkk + 1, " / ", self.ds.n_points_test)
            fig, axs = plt.subplots(self.x_dim)
            G = np.array(
                [
                    np.round(
                        self.ds.HMIN
                        + (self.gen_trajectories[kkk, it].T + 1)
                        * (self.ds.HMAX - self.ds.HMIN)
                        / 2
                    ).T
                    for it in range(self.ds.n_traj_per_point)
                ]
            )
            R = np.array(
                [
                    np.round(
                        self.ds.HMIN
                        + (self.ds.X_test_transp[kkk, it].T + 1)
                        * (self.ds.HMAX - self.ds.HMIN)
                        / 2
                    ).T
                    for it in range(self.ds.n_traj_per_point)
                ]
            )

            for d in range(self.x_dim):
                for traj_idx in range(n_trajs_to_plot):
                    axs[d].plot(tspan, R[traj_idx, d], color=self.colors[0])
                    axs[d].plot(tspan, G[traj_idx, d], color=self.colors[1])

            plt.tight_layout()
            fig.savefig(
                self.plots_path
                + "/"
                + self.model_name
                + "_Rescaled_Trajectories"
                + str(kkk)
                + ".png"
            )
            plt.close()

    def plot_histograms(self):
        bins = 50
        time_instant = -1
        print("Plotting histograms...")
        for kkk in range(self.ds.n_points_test):
            fig, ax = plt.subplots(self.x_dim, 1, figsize=(12, self.x_dim * 3))
            for d in range(self.x_dim):
                G = np.array(
                    [
                        np.round(
                            self.ds.HMIN
                            + (self.gen_trajectories[kkk, it].T + 1)
                            * (self.ds.HMAX - self.ds.HMIN)
                            / 2
                        ).T
                        for it in range(self.ds.n_traj_per_point)
                    ]
                )
                R = np.array(
                    [
                        np.round(
                            self.ds.HMIN
                            + (self.ds.X_test_transp[kkk, it].T + 1)
                            * (self.ds.HMAX - self.ds.HMIN)
                            / 2
                        ).T
                        for it in range(self.ds.n_traj_per_point)
                    ]
                )

                XXX = np.vstack((R[:, d, time_instant], G[:, d, time_instant])).T

                ax[d].hist(
                    XXX,
                    bins=bins,
                    stacked=False,
                    density=False,
                    color=self.colors,
                    label=self.leg,
                )
                ax[d].legend()
                ax[d].set_ylabel(self.species_names[d])

            figname = (
                self.plots_path
                + "/"
                + self.model_name
                + "_rescaled_hist_comparison_{}th_timestep_{}.png".format(
                    time_instant, kkk
                )
            )

            fig.savefig(figname)

            plt.close()

    def compute_wasserstein_distances(self):
        dist = np.zeros(shape=(self.ds.n_points_test, self.x_dim, self.traj_len))
        print("Computing and Plotting Wasserstein distances...")
        for kkk in range(self.ds.n_points_test):
            print("\tinit_state n = ", kkk)
            for m in range(self.x_dim):
                for t in range(self.traj_len):
                    A = self.ds.X_test_transp[kkk, :, m, t]
                    B = self.gen_trajectories[kkk, :, m, t]
                    dist[kkk, m, t] = wasserstein_distance(A, B)

        avg_dist = np.mean(dist, axis=0)
        markers = ["--", "-.", ":"]
        fig = plt.figure()
        for spec in range(self.x_dim):
            plt.plot(
                np.arange(self.traj_len),
                avg_dist[spec],
                markers[spec],
                label=self.species_names[spec],
            )
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("wass dist")
        plt.tight_layout()

        figname = (
            self.plots_path
            + "/"
            + self.model_name
            + "_Traj_avg_wass_distance_{}epochs_{}steps.png".format(
                self.n_epochs, self.traj_len
            )
        )
        fig.savefig(figname)
        distances_dict = {"gen_hist": B, "ssa_hist": A, "wass_dist": dist}
        file = open(self.plots_path + "/wgan_gp_distances.pickle", "wb")
        # dump information to that file
        pickle.dump(distances_dict, file)
        # close the file
        file.close()
