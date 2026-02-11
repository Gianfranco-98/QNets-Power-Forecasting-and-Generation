# Generic
import math

# Learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local files
from config import settings


# ------------------------------------------------------------------ #
# --------------------------- VAE Losses --------------------------- #
# ------------------------------------------------------------------ #
class VAELoss(nn.Module):
    """ Super class for VAE loss functions """

    def __init__(self):
        """ VAE Loss function initialization """
        super(VAELoss, self).__init__()
    
    def reconstruction_loss(self,
                            x: torch.Tensor,
                            y: torch.Tensor) -> torch.Tensor:
        """ Compute expected negative log likelihood 
        
        Parameters
        ----------
        - `x`: predicted (generated) variable
        - `y`: target variable

        Return
        ------
        computed reconstruction loss
        """
        return ((x - y) ** 2).sum(1).mean(0)

    def kl_divergence(self, distr_vars: torch.Tensor) -> torch.Tensor:
        """ Compute Kullback–Leibler divergence

        Parameters
        ----------
        `distr_var`: distribution variables

        Return
        ------
        `kl`: KL divergence value for a Gaussian distribution
        """
        mu_z, log_var_z = distr_vars
        kl = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
        kl = kl.sum(1).mean(0)
        return kl

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """ Loss computation
        
        Input
        -----
            - `x`: generated variable 
                   + mean and variance of the distribution
            - `y`: ground truth variable
        
        Output
        ------ 
        elbo_loss: computed VAE ELBO loss
        """
        y_gen, mu_z, log_var_z = x
        elbo_loss = (self.reconstruction_loss(y_gen, y)
                     + self.kl_divergence((mu_z, log_var_z)))
        return elbo_loss


class BetaDivergenceLoss(VAELoss):
    """ Beta-Divergence Loss from the original RVAE paper """

    def __init__(self, beta: float,
                 loss_std: float=settings["model"]["rvae"]["loss_std"]):
        """ Beta-Divergence Loss class initialization

        Parameters
        ----------
        - `beta`: RVAE Beta-Divergence Loss parameter
        - `loss_std`: std deviation constant for Gaussian loss
        """
        self.beta = beta
        self.loss_std = loss_std
        super(BetaDivergenceLoss, self).__init__()
    
    def reconstruction_loss(self,
                            y_gen: torch.Tensor,
                            y: torch.Tensor) -> torch.Tensor:
        """ Compute expected negative log likelihood

        Parameters
        ----------
        - `y_gen`: generated target variable
        - `y`: ground truth variable

        Return
        ------
        `rec_loss`: reconstruction loss value
        """
        if self.beta == 0:
            # Classical VAE reconstruction loss
            # NOTE: this is different from F.mse_loss
            rec_loss = VAELoss.reconstruction_loss(y_gen, y)
        else:
            # Assignments to simplify loss formula
            sigma = self.loss_std
            d = y_gen.shape[1]
            beta = self.beta
            pi = math.pi

            # Loss computation
            # TODO: check correctness with .mean(0)
            se = ((y_gen - y)**2).sum(1)
            rec_loss = -(((beta + 1) / beta)
                         * ((1 / ((2 * pi * (sigma**2)) ** (beta * d / 2)))
                            * torch.exp(-(beta / (2 * (sigma**2))) * se)
                            - 1))
            rec_loss = rec_loss.mean(0)
        return rec_loss


class QVAELoss(VAELoss):
    """ Class for QVAE loss functions """

    def __init__(self):
        """ QVAE Loss function initialization """
        super(QVAELoss, self).__init__()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """ Loss computation
        
        Input
        -----
            - `x`: generated variable
                   + mean and variance of the distribution
            - `y`: ground truth variable
        
        Output
        ------ 
        elbo_loss: computed QVAE ELBO loss
        """
        y_gen, mu_z, log_var_z = x
        elbo_loss = (self.reconstruction_loss(y_gen, y)
                     + 1.5 * self.kl_divergence((mu_z, log_var_z)).sum())
        return elbo_loss


class RQVAELoss(BetaDivergenceLoss):
    """ Class for QVAE loss functions """

    def __init__(self, beta: float,
                 loss_std: float=settings["model"]["rqvae"]["loss_std"]):
        """ QVAE Loss function initialization """
        super(RQVAELoss, self).__init__(beta, loss_std)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """ Loss computation
        
        Input
        -----
            - `x`: generated variable
                   + mean and variance of the distribution
            - `y`: ground truth variable
        
        Output
        ------ 
        elbo_loss: computed RQVAE ELBO loss
        """
        y_gen, mu_z, log_var_z = x
        elbo_loss = (self.reconstruction_loss(y_gen, y)
                     + self.kl_divergence((mu_z, log_var_z)).sum())
        return elbo_loss


# ------------------------------------------------------------------ #
# --------------------------- GAN Losses --------------------------- #
# ------------------------------------------------------------------ #
class GANLoss(nn.Module):
    """ Class for GAN loss functions """

    def __init__(self):
        """ GAN Loss function initialization """
        super(GANLoss, self).__init__()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """ Loss computation
        
        Input
        -----
            - `x`: GAN prediction (discriminator or generator)
            - `y`: real or fake data label
        
        Output
        ------ 
        gan_loss: computed GAN loss
        """
        return F.binary_cross_entropy(x, y)


class QGANLoss(nn.Module):
    """ Class for QGAN loss functions """

    def __init__(self):
        """ QGAN Loss function initialization """
        super(QGANLoss, self).__init__()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """ Loss computation
        
        Input
        -----
            - `x`: QGAN generated power
            - `y`: real power
        
        Output
        ------ 
        qgan_loss: computed QGAN loss
        """
        return F.binary_cross_entropy(x, y)

# ------------------------------------------------------------------ #
# --------------------------- RNN Losses --------------------------- #
# ------------------------------------------------------------------ #
class RNNLoss(nn.Module):
    """ Class for RNN loss functions """

    def __init__(self):
        """ RNN Loss function initialization """
        super(RNNLoss, self).__init__()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """ Loss computation
        
        Input
        -----
            - `x`: RNN generated power
            - `y`: real power
        
        Output
        ------ 
        rnn_loss: computed RNN MSE loss
        """
        rnn_loss = F.mse_loss(x, y)
        return rnn_loss


class QRNNLoss(nn.Module):
    """ Class for QRNN loss functions """

    def __init__(self):
        """ QRNN Loss function initialization """
        super(QRNNLoss, self).__init__()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """ Loss computation
        
        Input
        -----
            - `x`: QRNN generated power
            - `y`: real power
        
        Output
        ------ 
        qrnn_loss: computed QRNN L1 loss
        """
        qrnn_loss = F.l1_loss(x, y)
        return qrnn_loss