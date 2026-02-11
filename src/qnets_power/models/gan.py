# Generic
from typing import Tuple, Union
from collections import namedtuple

# Learning
import torch
import torch.nn as nn
from torchsummary import summary
import pytorch_lightning as pl

# Local files
from config import settings
from qnets_power.registry import register_model
from qnets_power.loss import GANLoss, QGANLoss
from qnets_power.models.quaternion_utils.quaternion_layers import \
    QuaternionLinear


# Define GAN net type
GANNet = namedtuple("GANNet", ["discriminator", "generator"])


@register_model("gan")
class GAN(pl.LightningModule):
    """ Generative Adversarial Network class (conditional version) """

    def __init__(self, in_size: int, in_cond: int,
                 latent_space: int=settings["model"]["gan"]["latent_space"],
                 learning_rate: float=settings["model"]["gan"]["learning_rate"],
                 weight_decay: float=settings["model"]["gan"]["weight_decay"],
                 net: Union[GANNet, None]=None, loss_fun: Union[GANLoss, None]=None,
                 name: str="gan", device: torch.device=settings["def_device"]):
        """ GAN class initialization

        Parameters
        ----------
        - `in_size`: size of the input variable to reconstruct
        - `in_cond`: size of the conditional input
        - `latent_space`: dimension of the latent space
        - `learning_rate`: learning rate for the GAN optimizer
        - `weight_decay`: weight_decay for the GAN optimizer
        - `net`: eventual GAN neural network
        - `loss_fun`: eventual GAN loss function
        - `name`: name of the network
        - `device`: target device of the GAN loss and modules
        """
        # Basic initialization
        super(GAN, self).__init__()
        self.name = name
        self.in_size = in_size
        self.in_cond = in_cond
        self.latent_space = latent_space
        self.automatic_optimization = False

        # Network initialization
        self.in_dis_features = in_size + in_cond
        self.out_dis_features = 1
        self.in_gen_features = latent_space + in_cond
        self.out_gen_features = in_size

        # Training initialization
        self.tgt_device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = GANLoss().to(device) if loss_fun is None else loss_fun

        # ----------------------------------------------------------- #
        # ---------------------| GAN Structure |--------------------- #
        # ----------------------------------------------------------- #
        if net is None:
            self.discriminator = nn.Sequential(
                nn.Linear(self.in_dis_features, 400),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(400, 200),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(200, self.out_dis_features),
                nn.Sigmoid())
            self.generator = nn.Sequential(
                nn.Linear(self.in_gen_features, 200),
                nn.ReLU(),
                nn.Linear(200, 400),
                nn.ReLU(),
                nn.Linear(400, self.out_gen_features),
                nn.Tanh())
        else:
            self.discriminator = net.discriminator
            self.generator = net.generator
        self.weights_init()
        # ----------------------------------------------------------- #

    def weights_init(self, mean: float=0.0, std: float=0.02):
        """ Normal distribution initialization

        Parameters
        ----------
        - `mean`: mean of the distribution
        - `std`: standard deviation of the distribution
        """
        # TODO: check why init the discriminator worsen performances
        """for l in self.discriminator:
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight.data, mean=mean, std=std)"""
        for l in self.generator:
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight.data, mean=mean, std=std)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """ Training step of the model

        Parameters
        ----------
        - `batch`: batch of data
        - `batch_idx`: index of the current batch (0 -> len(dataset))
        """
        # Training initialization
        x, y = batch
        dis_opt, gen_opt = self.optimizers()
        noise = torch.randn(
            2, x.shape[0], self.latent_space).to(self.tgt_device)
        real_label = torch.ones(x.shape[0]).to(self.tgt_device)
        fake_label = torch.zeros(x.shape[0]).to(self.tgt_device)

        # Train discriminator
        dis_opt.zero_grad()
        # 1 - Real data input
        data_prob = self(x, y, net="dis")
        dis_loss_real = self.loss(data_prob, real_label).max()
        # 2 - Fake data input
        fake_data = self(x, noise[0], net="gen")
        data_prob = self(x, fake_data, net="dis")
        dis_loss_fake = self.loss(data_prob, fake_label).max()
        dis_loss = dis_loss_real + dis_loss_fake
        self.manual_backward(dis_loss)
        dis_opt.step()

        # Train generator
        gen_opt.zero_grad()
        fake_data = self(x, noise[1], net="gen")
        data_prob = self(x, fake_data, net="dis")
        gen_loss = self.loss(data_prob, real_label).max()
        self.manual_backward(gen_loss)
        gen_opt.step()
        
        # Log
        self.log_dict({"train_dis_loss": dis_loss.item(), 
                       "train_gen_loss": gen_loss.item()}, 
                      prog_bar=True, on_step=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """ Validation step of the model

        Parameters
        ----------
        - `batch`: batch of data
        - `batch_idx`: index of the current batch (0 -> len(dataset))
        """
        with torch.no_grad():
            # Validation initialization
            x, y = batch
            noise = torch.randn(
                2, x.shape[0], self.latent_space).to(self.tgt_device)
            real_label = torch.ones(x.shape[0]).to(self.tgt_device)
            fake_label = torch.zeros(x.shape[0]).to(self.tgt_device)

            # Evaluate discriminator
            # 1 - Real data input
            data_prob = self(x, y, net="dis")
            dis_loss_real = self.loss(data_prob, real_label).max()
            # 2 - Fake data input
            fake_data = self(x, noise[0], net="gen")
            data_prob = self(x, fake_data, net="dis")
            dis_loss_fake = self.loss(data_prob, fake_label).max()
            dis_loss = dis_loss_real + dis_loss_fake

            # Evaluate generator
            fake_data = self(x, noise[1], net="gen")
            data_prob = self(x, fake_data, net="dis")
            gen_loss = self.loss(data_prob, real_label).max()
            
        # Log
        self.log_dict({"val_dis_loss": dis_loss.item(), 
                       "val_gen_loss": gen_loss.item(),
                       "step": self.current_epoch}, 
                      prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Adam:
        """ Set the optimizer for the model """
        dis_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=[0.0, 0.9],
            weight_decay=self.weight_decay)
        gen_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate, betas=[0.0, 0.9],
            weight_decay=self.weight_decay)
        return dis_opt, gen_opt

    def summary(self):
        """ Make a summary of the model """
        print("-----------------------------------------------------")
        print(f"------------------- {self.name} summary -------------------")
        print("-----------------------------------------------------")
        print("_________ Discriminator _________")
        summary(self.discriminator, input_size=(1, self.in_dis_features))
        print("_________ Generator _________")
        summary(self.generator, input_size=(1, self.in_gen_features))

    def generate(self, x_cond: torch.Tensor, n_samples: int=1
                 ) -> torch.Tensor:
        """ Generate a sample of the learned variable 
        
        Parameters
        ----------
        - `x_cond`: conditional input
        - `n_samples`: number of samples to generate

        Return
        ------
        `sample`: generated sample
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_space).to(self.tgt_device)
            x_cond = x_cond.repeat(n_samples).view(n_samples, self.in_cond)
            sample = self(x_cond, z, net="gen").view(n_samples, -1)
        return sample

    def forward(self, x_cond: torch.Tensor, x: torch.Tensor,
                net: str="gen") -> torch.Tensor:
        """ GAN complete inference 
        
        Input:
            - `x_cond`: conditional input data
            - `x`: input (real or fake) or noise
            - `net`: "dis" for discriminator and "gen" for generator

        Output:
            - `out`: output from the network
        """
        x = torch.cat((x, x_cond), dim=1)
        if net == "dis":
            out = self.discriminator(x)
        elif net == "gen":
            out = self.generator(x)
        return out.squeeze()
    
    @staticmethod
    def get_default_kwargs(pow_hours: int, in_cond: int,
                           device: torch.device=settings["def_device"]) -> dict:
        """ Return required base args for the model.
        This is used to have a unified model instantiation method.
        
        Parameters
        ----------
        - `pow_hours`: number of hours with an actual power production
        - `in_cond`: size of the conditional input
        - `device`: target device of the VAE loss and modules
        
        Return
        ------
        dictionary with fields:
            - `in_size`: size of the input variable to reconstruct
            - `in_cond`: size of the conditional input
            - `device`: target device of the VAE loss and modules
        """
        return {
            "in_size": pow_hours,
            "in_cond": in_cond,
            "device": device}


@register_model("qgan")
class QGAN(GAN):
    """ Quaternion GAN class (conditional version) """

    def __init__(self, in_size: int, in_cond: int,
                 latent_space: int=settings["model"]["qgan"]["latent_space"],
                 learning_rate: float=settings["model"]["qgan"]["learning_rate"],
                 weight_decay: float=settings["model"]["qgan"]["weight_decay"],
                 loss_fun: Union[GANLoss, None]=None, name: str="qgan",
                 device: torch.device=settings["def_device"]):
        """ QGAN class initialization

        Parameters
        ----------
        - `in_size`: size of the input variable to reconstruct
        - `in_cond`: size of the conditional input
        - `latent_space`: dimension of the latent space
        - `learning_rate`: learning rate for the QGAN optimizer
        - `weight_decay`: weight_decay for the QGAN optimizer
        - `loss_fun`: eventual loss function
        - `name`: name of the network
        - `device`: target device of the QGAN loss and modules
        """
        # Loss initialization
        loss_fun = QGANLoss() if loss_fun is None else loss_fun

        # Network initialization
        in_dis_features = in_size + in_cond
        out_dis_features = 1
        in_gen_features = 4 * (latent_space + in_cond)
        out_gen_features = in_size

        # ---------------------------------------------------------- #
        # --------------------| QGAN Structure |-------------------- #
        # ---------------------------------------------------------- #
        discriminator = nn.Sequential(
            QuaternionLinear(in_dis_features, 400),
            nn.LeakyReLU(negative_slope=0.2),
            QuaternionLinear(400, 200),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(start_dim=1),
            QuaternionLinear(200 * 4, 4),
            nn.Linear(4, 1),
            nn.Sigmoid())
        generator = nn.Sequential(
            QuaternionLinear(in_gen_features, 200),
            nn.ReLU(),
            QuaternionLinear(200, 400),
            nn.ReLU(),
            QuaternionLinear(400, out_gen_features),
            nn.Tanh())
        net = GANNet(discriminator, generator)
        # ---------------------------------------------------------- #

        # Super initialization
        super(QGAN, self).__init__(
            in_size, in_cond, latent_space, learning_rate,
            weight_decay, net, loss_fun, name, device)
        self.in_cond = in_cond
        self.in_dis_features = in_dis_features
        self.out_dis_features = out_dis_features
        self.in_gen_features = in_gen_features
        self.out_gen_features = out_gen_features

    def weights_init(self, mean: float=0.0, std: float=0.02):
        """ Normal distribution initialization

        Parameters
        ----------
        - `mean`: mean of the distribution
        - `std`: standard deviation of the distribution
        """
        # TODO: check why init the discriminator worsen performances
        """for l in self.discriminator:
            if isinstance(l, QuaternionLinear):
                nn.init.normal_(l.r_weight.data, mean, std)
                nn.init.normal_(l.i_weight.data, mean, std)
                nn.init.normal_(l.j_weight.data, mean, std)
                nn.init.normal_(l.k_weight.data, mean, std)"""
        for l in self.generator:
            if isinstance(l, QuaternionLinear):
                nn.init.normal_(l.r_weight.data, mean, std)
                nn.init.normal_(l.i_weight.data, mean, std)
                nn.init.normal_(l.j_weight.data, mean, std)
                nn.init.normal_(l.k_weight.data, mean, std)

    def generate(self, x_cond: torch.Tensor, n_samples: int=1
                 ) -> torch.Tensor:
        """ Generate a sample of the learned variable 

        Parameters
        ----------
        - `x_cond`: conditional input
        - `n_samples`: number of samples to generate

        Return
        ------
        `sample`: generated sample
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_space).to(self.tgt_device)
            x_cond = x_cond.repeat(n_samples, 1).view(n_samples, -1)
            sample = self(x_cond, z, net="gen").view(n_samples, -1)
        return sample

    def forward(self, x_cond: torch.Tensor, x: torch.Tensor,
                net: str="gen") -> torch.Tensor:
        """ GAN complete inference 
        
        Input:
            - `x_cond`: conditional input data
            - `x`: input (real or fake) or noise
            - `net`: "dis" for discriminator and "gen" for generator

        Output:
            - `out`: output from the network
        """
        if net == "dis":
            x = x.repeat(1, 4).view(x.shape[0], 4, -1)
            x = torch.cat((x, x_cond), dim=2)
        else:
            noise = torch.randn(
                3, x.shape[0], self.latent_space).to(self.tgt_device)
            x = torch.stack([x, noise[0], noise[1], noise[2]], dim=1)
            x = torch.cat((x.flatten(1), x_cond.flatten(1)), dim=1)
        if net == "dis":
            out = self.discriminator(x)
        elif net == "gen":
            out = self.generator(x)
        return out.squeeze()