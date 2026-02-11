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
from qnets_power.loss import RNNLoss, QRNNLoss
from qnets_power.models.quaternion_utils.quaternion_layers_2 import \
    QuaternionLinearAutograd


# Define RNN net type
RNNNet = namedtuple("RNNNet", ["gru", "fc"])


@register_model("rnn")
class RNN(pl.LightningModule):
    """ Recurrent Neural Network class """

    def __init__(self, in_size: int, out_size: int,
                 hidden_size: int=settings["model"]["rnn"]["hidden_size"],
                 num_gru_layers: int=settings["model"]["rnn"]["num_gru_layers"], 
                 dropout_prob: float=settings["model"]["rnn"]["dropout_prob"],
                 learning_rate: float=settings["model"]["rnn"]["learning_rate"],
                 weight_decay: float=settings["model"]["rnn"]["weight_decay"],
                 net: Union[RNNNet, None]=None, loss_fun: Union[RNNLoss, None]=None,
                 name: str="rnn", device: torch.device=settings["def_device"]):
        """ RNN class initialization

        Parameters
        ----------
        - `in_size`: size of the input variables
        - `out_size`: size of the variable to forecast
        - `hidden_size`: size of the RNN hidden layer
        - `num_gru_layers`: size of the multilayer GRU
        - `dropout_prob`: probability of the GRU dropout
        - `learning_rate`: learning rate for the RNN optimizer
        - `weight_decay`: weight_decay for the RNN optimizer
        - `net`: eventual RNN neural network
        - `loss_fun`: eventual RNN loss function
        - `name`: name of the network
        - `device`: target device of the RNN loss and modules
        """
        # Basic initialization
        super(RNN, self).__init__()
        self.name = name
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.dropout_prob = dropout_prob

        # Training initialization
        self.tgt_device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = RNNLoss().to(device) if loss_fun is None else loss_fun

        # ----------------------------------------------------------- #
        # ---------------------| RNN Structure |--------------------- #
        # ----------------------------------------------------------- #
        if net is None:
            self.gru = nn.GRU(in_size, hidden_size, num_layers=num_gru_layers,
                              batch_first=True, dropout=dropout_prob)
            self.fc = nn.Linear(hidden_size, out_size)
        else:
            self.gru = net.gru
            self.fc = net.fc
        # ----------------------------------------------------------- #

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """ Training step of the RNN

        Parameters
        ----------
        - `batch`: batch of data
        - `batch_idx`: index of the current batch (0 -> len(dataset))
        """
        # Data -> Inference -> Loss -> Log
        x, y = batch
        out = self(x)
        loss = self.loss(out, y).max()
        self.log("train_loss", loss.item(), on_step=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """ Validation step of the RNN

        Parameters
        ----------
        - `batch`: batch of data
        - `batch_idx`: index of the current batch (0 -> len(dataset))
        """
        # Data -> Inference -> Loss -> Log
        with torch.no_grad():
            x, y = batch
            out = self(x)
            loss = self.loss(out, y).max()
        self.log_dict({"val_loss": loss.item(), "step": self.current_epoch},
                      on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """ Set the optimizer for the RNN """
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

    def summary(self):
        """ Make a summary of the RNN """
        print("-----------------------------------------------------")
        print(f"------------------- {self.name} summary -------------------")
        print("-----------------------------------------------------")
        summary(self, input_size=(1, self.in_size))

    def generate(self, x: torch.Tensor, n_samples: int=1
                 ) -> torch.Tensor:
        """ Generate a sample of the learned variable 

        Parameters
        ----------
        - `x`: input variables
        - `n_samples`: number of samples to generate

        Return
        ------
        `sample`: generated sample
        """
        with torch.no_grad():
            sample = self(x.unsqueeze(0), with_grads=False).unsqueeze(0)
        return sample

    def forward(self, x: torch.Tensor, with_grads: bool=True
                ) -> torch.Tensor:
        """ RNN complete inference 
        
        Input:
            - `x`: input tensor
            - `with_grads`: True if to compute grads

        Output:
            - `out`: forecasted variable
        """
        start_context = torch.zeros(
            self.num_gru_layers, x.shape[0], 
            self.hidden_size).to(self.tgt_device)
        if with_grads:
            start_context = start_context.requires_grad_()
        # TODO: check for out[:, -1, :] from GRU
        # NOTE: the GRU automatically reinject historical data
        out, context = self.gru(x.unsqueeze(1), start_context)
        # out = out[:, -1, :]
        if self.fc is not None:
            out = self.fc(out)
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
            - `out_size`: size of the variable to forecast
            - `device`: target device of the RNN loss and modules
        """
        return {
            "in_size": in_cond,
            "out_size": pow_hours,
            "device": device}


@register_model("qrnn")
class QRNN(RNN):
    """ Quaternion Recurrent Neural Network class """

    def __init__(self, in_size: int, out_size: int,
                 hidden_size: int=settings["model"]["qrnn"]["hidden_size"],
                 num_gru_layers: int=settings["model"]["qrnn"]["num_gru_layers"], 
                 dropout_prob: float=settings["model"]["qrnn"]["dropout_prob"],
                 learning_rate: float=settings["model"]["qrnn"]["learning_rate"],
                 weight_decay: float=settings["model"]["qrnn"]["weight_decay"],
                 loss_fun: Union[RNNLoss, None]=None, name: str="qrnn",
                 device: torch.device=settings["def_device"]):
        """ QRNN class initialization

        Parameters
        ----------
        - `in_size`: size of the input variables
        - `out_size`: size of the variable to forecast
        - `hidden_size`: size of the QRNN hidden layer
        - `num_gru_layers`: size of the multilayer GRU
        - `dropout_prob`: probability of the GRU dropout
        - `learning_rate`: learning rate for the QRNN optimizer
        - `weight_decay`: weight_decay for the QRNN optimizer
        - `loss_fun`: eventual QRNN loss function
        - `name`: name of the network
        - `device`: target device of the QRNN loss and modules
        """
        # Basic initialization
        self.dropout_prob = dropout_prob
        name = f"qgru_{num_gru_layers}"

        # Loss initialization
        loss = QRNNLoss().to(device) if loss_fun is None else loss_fun

        # ---------------------------------------------------------- #
        # --------------------| QRNN Structure |-------------------- #
        # ---------------------------------------------------------- #
        qgru = QGRU(in_size // 4, hidden_size // 4,
                    num_layers=num_gru_layers, 
                    dropout=dropout_prob, device=device)
        fc = nn.Linear(hidden_size, out_size)
        net = RNNNet(qgru, fc)
        # ---------------------------------------------------------- #

        # Super initialization
        super(QRNN, self).__init__(
            in_size, out_size, hidden_size, num_gru_layers, dropout_prob,
            learning_rate, weight_decay, net, loss, name, device)

    def configure_optimizers(self) -> torch.optim.Adam:
        """ Set the optimizer for the RNN """
        return torch.optim.Adamax(
            self.parameters(), lr=self.learning_rate,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=self.weight_decay)

    def generate(self, x: torch.Tensor, n_samples: int=1
                 ) -> torch.Tensor:
        """ Generate a sample of the learned variable 

        Parameters
        ----------
        - `x`: input variables
        - `n_samples`: number of samples to generate

        Return
        ------
        `sample`: generated sample
        """
        with torch.no_grad():
            sample = self(x.unsqueeze(0), with_grads=False)
        return sample

    def forward(self, x: torch.Tensor, with_grads: bool=True
                ) -> torch.Tensor:
        """ QRNN complete inference 
        
        Input:
            - `x`: input tensor
            - `with_grads`: True if to compute grads

        Output:
            - `out`: forecasted variable
        """
        out = self.gru(x)
        if self.fc is not None:
            out = self.fc(out)
        return out#.squeeze()


@register_model("qgru")
class QGRU(nn.Module):
    """ Quaternion Gated Recurrent Unit implementation """

    def __init__(self, input_dim: int, hidden_dim: int, 
                 num_layers: int, dropout: int, device: torch.device):
        """ QGRU class initialization

        Parameters
        ----------
        - `input_dim`: size of the input variables
        - `hidden_dim`: size of the QGRU hidden layer
        - `num_layers`: size of the multilayer GRU
        - `dropout`: probability of the GRU dropout
        - `device`: target device var for the QGRU context variables
        """
        # Basic initialization
        super(QGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.tgt_device = device
        
        # ---------------------------------------------------------- #
        # --------------------| QGRU Structure |-------------------- #
        # ---------------------------------------------------------- #
        self.wf_list, self.uf_list = [], []
        for l in range(num_layers):
            self.wf_list.append(QuaternionLinearAutograd(
                input_dim, self.hidden_dim*3, bias=True))
            self.uf_list.append(QuaternionLinearAutograd(
                self.hidden_dim, self.hidden_dim*3, bias=True))
            input_dim = self.hidden_dim
        self.wf = nn.ModuleList(self.wf_list)
        self.uf = nn.ModuleList(self.uf_list)
        self.act = nn.ReLU()
        self.act_gate = nn.Sigmoid()
        self.dropout = nn.Dropout(p=float(dropout))
        # ---------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ QGRU complete inference 
        
        Input:
            - `x`: input tensor

        Output:
            - `out`: QGRU hidden state
        """
        # Hidden state initialization
        # TODO: find a way to solve autograd with tensor indexing
        h = torch.zeros(
            x.shape[0], self.hidden_dim*4).to(
                self.tgt_device).requires_grad_()
        if self.num_layers > 1:
            h2 = torch.zeros(
                x.shape[0], self.hidden_dim*4).to(
                    self.tgt_device).requires_grad_()
        if self.num_layers > 2:
            h3 = torch.zeros(
                x.shape[0], self.hidden_dim*4).to(
                    self.tgt_device).requires_grad_()

        # Stack in each time step
        for k in range(x.shape[1]):
            x_ = x[:,k,:]
            in_gru = x_
            for gru_idx in range(self.num_layers):
                if gru_idx == 1:
                    h = h2
                elif gru_idx == 2:
                    h = h3
                h = self.dropout(h)
                wxf, wxi, wxn = (self.wf[gru_idx](in_gru, self.dropout_prob))
                uxf, uxi, uxn = (self.uf[gru_idx](h, self.dropout_prob))
                rt, zt = self.act_gate(wxf + uxf), self.act_gate(wxi + uxi)
                nt = self.act(wxn + (rt * uxn))
                h = zt * h + (1 - zt) * nt
                in_gru = h

        return h