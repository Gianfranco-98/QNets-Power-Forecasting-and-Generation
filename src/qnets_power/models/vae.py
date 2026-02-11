# Generic
from typing import Tuple, Union
from collections import namedtuple

# Learning
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchsummary import summary

# Local files
from config import settings
from qnets_power.registry import register_model
from qnets_power.loss import \
    VAELoss, BetaDivergenceLoss, QVAELoss, RQVAELoss
from qnets_power.models.quaternion_utils.quaternion_layers import \
    QuaternionLinear


# Define VAE net type
VAENet = namedtuple("VAENet", ["encoder", "decoder"])


@register_model("vae")
class VAE(pl.LightningModule):
    """ Variational Autoencoder class (conditional version) """

    def __init__(self, in_size: int, in_cond: int,
                 latent_space: int=settings["model"]["vae"]["latent_space"],
                 learning_rate: float=settings["model"]["vae"]["learning_rate"],
                 weight_decay: float=settings["model"]["vae"]["weight_decay"],
                 net: Union[VAENet, None]=None, loss_fun: Union[VAELoss, None]=None,
                 name: str="vae", device: torch.device=settings["def_device"]):
        """ VAE class initialization

        Parameters
        ----------
        - `in_size`: size of the input variable to reconstruct
        - `in_cond`: size of the conditional input
        - `latent_space`: dimension of the latent space
        - `learning_rate`: learning rate for the VAE optimizer
        - `weight_decay`: weight_decay for the VAE optimizer
        - `net`: eventual VAE neural network
        - `loss_fun`: eventual VAE loss function
        - `name`: name of the network
        - `device`: target device of the VAE loss and modules
        """
        # Basic initialization
        super(VAE, self).__init__()
        self.name = name
        self.in_size = in_size
        self.in_cond = in_cond
        self.latent_space = latent_space

        # Network initialization
        self.in_enc_features = in_size + in_cond
        self.out_enc_features = 2 * latent_space
        self.in_dec_features = latent_space + in_cond
        self.out_dec_features = in_size

        # Training initialization
        self.tgt_device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = VAELoss().to(device) if loss_fun is None else loss_fun

        # ----------------------------------------------------------- #
        # ---------------------| VAE Structure |--------------------- #
        # ----------------------------------------------------------- #
        if net is None:
            self.encoder = nn.Sequential(
                nn.Linear(self.in_enc_features, 200),
                nn.ReLU(),
                nn.Linear(200, 400),
                nn.ReLU(),
                nn.Linear(400, self.out_enc_features))
            self.decoder = nn.Sequential(
                nn.Linear(self.in_dec_features, 400),
                nn.ReLU(),
                nn.Linear(400, 200),
                nn.ReLU(),
                nn.Linear(200, self.out_dec_features))
        else:
            self.encoder = net.encoder
            self.decoder = net.decoder
        # ----------------------------------------------------------- #

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """ Training step of the model

        Parameters
        ----------
        - `batch`: batch of data
        - `batch_idx`: index of the current batch (0 -> len(dataset))
        """
        # Data -> Inference -> Loss -> Log
        x, y = batch
        y_gen, mu_z, log_var_z = self(x, y)
        loss = self.loss((y_gen, mu_z, log_var_z), y).max()
        self.log("train_loss", loss.item(), on_step=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """ Validation step of the model

        Parameters
        ----------
        - `batch`: batch of data
        - `batch_idx`: index of the current batch (0 -> len(dataset))
        """
        # Data -> Inference -> Loss -> Log
        with torch.no_grad():
            x, y = batch
            y_gen, mu_z, log_var_z = self(x, y)
            loss = self.loss((y_gen, mu_z, log_var_z), y).max()
        self.log_dict({"val_loss": loss.item(), "step": self.current_epoch},
                      on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """ Set the optimizer for the model """
        return torch.optim.Adam(self.parameters(), 
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)

    def summary(self):
        """ Make a summary of the model """
        print("-----------------------------------------------------")
        print(f"------------------- {self.name} summary -------------------")
        print("-----------------------------------------------------")
        print("_________ Encoder _________")
        summary(self.encoder, input_size=(1, self.in_enc_features))
        print("_________ Decoder _________")
        summary(self.decoder, input_size=(1, self.in_dec_features))

    def encode(self, x_cond: torch.Tensor, x: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Encode conditional input and tgt variable into latent space
        
        Parameters
        ----------
        - `x_cond`: conditional input
        - `x`: random variable to fit

        Return
        ------
        - `mu_z`: mean of the latent distribution
        - `log_var_z`: variance of the latent distribution
        """
        enc_input = torch.cat((x, x_cond), 1)
        mu_var_z = self.encoder(enc_input)
        mu_z, log_var_z = torch.split(mu_var_z, self.latent_space, 1)
        return mu_z, log_var_z

    def decode(self, x_cond: torch.Tensor, z: torch.Tensor
               ) -> torch.Tensor:
        """ Decode latent space into target variable
        
        Parameters
        ----------
        - `x_cond`: conditional input
        - `z`: sample from the latent space

        Return
        ------
        `out`: reconstructed target variable
        """
        dec_input = torch.cat((z, x_cond), 1)
        out = self.decoder(dec_input)
        return out

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
            sample = self.decode(x_cond, z).view(n_samples, -1)
        return sample

    def forward(self, x_cond: torch.Tensor, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ VAE complete inference 
        
        Input:
            - `x_cond`: conditional input data
            - `x`: random variable to fit

        Output:
            - `out`: reconstructed variable
            - `mu_z`: mean of the latent distribution
            - `log_var_z`: variance of the latent distribution
        """
        mu_z, log_var_z = self.encode(x_cond, x)
        z = VAE.reparametrize(mu_z, log_var_z)
        out = self.decode(x_cond, z)
        return out, mu_z, log_var_z

    @staticmethod
    def reparametrize(mu: torch.Tensor, log_var: torch.Tensor
                      ) -> torch.Tensor:
        """ Apply VAE reparametrization trick

        Parameters
        ----------
        - `mu`: predicted mean of the latent distribution
        - `log_var`: predicted log variance of the latent distribution

        Return
        ------
        `z`: gaussian sample scaled by the latent distribution params
        """
        # TODO: test with std_dev from enc output instead of variance
        std = torch.exp((1/2) * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
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


@register_model("rvae")
class RVAE(VAE):
    """ Robust VAE class (conditional version) """

    def __init__(self, in_size: int, in_cond: int,
                 beta: float=settings["model"]["rvae"]["beta"],
                 loss_std: float=settings["model"]["rvae"]["loss_std"],
                 latent_space: int=settings["model"]["rvae"]["latent_space"],
                 learning_rate: float=settings["model"]["rvae"]["learning_rate"],
                 weight_decay: float=settings["model"]["rvae"]["weight_decay"],
                 name: str="rvae", device: torch.device=settings["def_device"]):
        """ VAE class initialization

        Parameters
        ----------
        - `in_size`: size of the input variable to reconstruct
        - `in_cond`: size of the conditional input
        - `beta`: RVAE Beta-Divergence Loss parameter
        - `loss_std`: standard deviation of the rqvae loss
        - `latent_space`: dimension of the latent space
        - `learning_rate`: learning rate for the RVAE optimizer
        - `weight_decay`: weight_decay for the RVAE optimizer
        - `name`: name of the network
        - `device`: target device of the RVAE loss and modules
        """
        # Basic initialization
        loss_fun = BetaDivergenceLoss(beta, loss_std).to(device)
        super(RVAE, self).__init__(
            in_size, in_cond, latent_space,
            learning_rate, weight_decay, None, loss_fun, name, device)


@register_model("qvae")
class QVAE(VAE):
    """ Quaternion VAE class (conditional version) """

    def __init__(self, in_size: int, in_cond: int,
                 latent_space: int=settings["model"]["qvae"]["latent_space"],
                 learning_rate: float=settings["model"]["qvae"]["learning_rate"],
                 weight_decay: float=settings["model"]["qvae"]["weight_decay"],
                 loss_fun: Union[VAELoss, None]=None, name: str="qvae",
                 device: torch.device=settings["def_device"]):
        """ QVAE class initialization

        Parameters
        ----------
        - `in_size`: size of the input variable to reconstruct
        - `in_cond`: size of the conditional input
        - `latent_space`: dimension of the latent space
        - `learning_rate`: learning rate for the QVAE optimizer
        - `weight_decay`: weight_decay for the QVAE optimizer
        - `loss_fun`: eventual loss function
        - `name`: name of the network
        - `device`: target device of the QVAE loss and modules
        """
        # Loss initialization
        loss_fun = QVAELoss() if loss_fun is None else loss_fun

        # Network initialization
        in_enc_features = in_size + in_cond
        out_enc_features = 8 * latent_space
        in_dec_features = 4 * (latent_space + in_cond)
        out_dec_features = in_size

        # ---------------------------------------------------------- #
        # --------------------| QVAE Structure |-------------------- #
        # ---------------------------------------------------------- #
        # TODO: check input and output features
        encoder = nn.Sequential(
            QuaternionLinear(in_enc_features, 200),
            nn.ReLU(),
            QuaternionLinear(200, 400),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            QuaternionLinear(400 * 4, out_enc_features))
        decoder = nn.Sequential(
            QuaternionLinear(in_dec_features, 200),
            nn.ReLU(),
            QuaternionLinear(200, 400),
            nn.ReLU(),
            QuaternionLinear(400, out_dec_features))
        net = VAENet(encoder, decoder)
        # ---------------------------------------------------------- #

        # Super initialization
        super(QVAE, self).__init__(
            in_size, in_cond, latent_space, learning_rate,
            weight_decay, net, loss_fun, name, device)
        self.in_enc_features = in_enc_features
        self.out_enc_features = out_enc_features
        self.in_dec_features = in_dec_features
        self.out_dec_features = out_dec_features

    def encode(self, x_cond: torch.Tensor, x: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Encode conditional input and tgt variable into latent space

        Parameters
        ----------
        - `x_cond`: conditional input
        - `x`: random variable to fit

        Return
        ------
        - `mu_z`: mean of the latent distribution
        - `log_var_z`: variance of the latent distribution
        """
        x = x.repeat(1, 4).view(x.shape[0], 4, -1)
        enc_input = torch.cat((x, x_cond), 2)
        mu_var_z = self.encoder(enc_input)
        mu_z, log_var_z = torch.split(mu_var_z, self.latent_space * 4, 1)
        mu_z = mu_z.view(mu_z.shape[0], 4, self.latent_space)
        log_var_z = log_var_z.view(log_var_z.shape[0], 4, self.latent_space)
        return mu_z, log_var_z

    def decode(self, x_cond: torch.Tensor, z: torch.Tensor
               ) -> torch.Tensor:
        """ Decode latent space into target variable
        
        Parameters
        ----------
        - `x_cond`: conditional input
        - `z`: sample from the latent space

        Return
        ------
        `out`: reconstructed target variable
        """
        dec_input = torch.cat((z, x_cond), 1)
        out = self.decoder(dec_input)
        return out

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
            x_cond = x_cond.repeat(n_samples, 1).view(n_samples, -1)
            z = torch.randn(
                n_samples, self.latent_space * 4).to(self.tgt_device)
            sample = self.decode(x_cond, z).view(n_samples, -1)
        return sample

    def forward(self, x_cond: torch.Tensor, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ QVAE complete inference 

        Input:
            - `x_cond`: conditional input data
            - `x`: random variable to fit

        Output:
            - `out`: reconstructed variable
            - `mu_z`: mean of the latent distribution
            - `log_var_z`: variance of the latent distribution
        """
        mu_z, log_var_z = self.encode(x_cond, x)
        z = VAE.reparametrize(mu_z, log_var_z)
        out = self.decode(
            x_cond.view(x_cond.shape[0], -1), z.view(z.shape[0], -1))
        return out, mu_z, log_var_z


@register_model("rqvae")
class RQVAE(QVAE):
    """ Robust Quaternion VAE class (conditional version) """

    def __init__(self, in_size: int, in_cond: int,
                 beta: float=settings["model"]["rqvae"]["beta"],
                 loss_std: float=settings["model"]["rqvae"]["loss_std"],
                 latent_space: int=settings["model"]["rqvae"]["latent_space"],
                 learning_rate: float=settings["model"]["rqvae"]["learning_rate"],
                 weight_decay: float=settings["model"]["rqvae"]["weight_decay"],
                 name: str="rqvae", device: torch.device=settings["def_device"]):
        """ QVAE class initialization

        Parameters
        ----------
        - `in_size`: size of the input variable to reconstruct
        - `in_cond`: size of the conditional input
        - `beta`: RQVAE Beta-Divergence Loss parameter
        - `loss_std`: standard deviation of the rqvae loss
        - `latent_space`: dimension of the latent space
        - `learning_rate`: learning rate for the RQVAE optimizer
        - `weight_decay`: weight_decay for the RQVAE optimizer
        - `name`: name of the network
        - `device`: target device of the RQVAE loss and modules
        """
        # Basic initialization
        loss_fun = RQVAELoss(beta, loss_std).to(device)
        super(RQVAE, self).__init__(
            in_size, in_cond, latent_space, learning_rate,
            weight_decay, loss_fun, name, device)