# Generic
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_train_data(tr_iterations: int, val_iterations: int, epoches: int,
                    tr_losses: np.ndarray, val_losses: np.ndarray):
    """ Plot a graph with the training trend [debug-only]
  
    Parameters
    ----------
    - `tr_iterations`: number of iterations for each epoch [train]
    - `val_iterations`: number of iterations for each epoch [val]
    - `epoches`: actual epoch number (starting from 1)
    - `tr_losses`: array of loss values [train]
    - `val_losses`: array of loss values [val]
    """
    # Data preparation
    train_iterations_list = list(range(epoches*(tr_iterations)))
    val_iterations_list = list(range(epoches*(val_iterations)))
    epoches_list = list(range(epoches))

    # Adjust validation array dimension
    val_error = len(val_losses) - len(val_iterations_list)
    if val_error > 0:
        val_losses = val_losses[:-val_error]

    # Per-iteration plot
    fig = plt.figure()
    plt.title("Per-iteration Loss [train]")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    l1, = plt.plot(train_iterations_list, tr_losses, c="blue")
    plt.legend(handles=[l1], labels=["Train loss"], loc="best")
    plt.show()
    fig = plt.figure()
    plt.title("Per-iteration Loss [val]")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    l2, = plt.plot(val_iterations_list, val_losses, c="red")
    plt.legend(handles=[l2], labels=["Validation loss"], loc="best")
    plt.show()

    # Per-epoch plot
    fig = plt.figure()
    plt.title("Per-epoch Loss")
    plt.xlabel("Epoches")
    plt.ylabel("Value")
    train_avg_losses = [np.array(tr_losses[i:i+tr_iterations]).mean() 
                        for i in range(0, len(tr_losses), tr_iterations)]
    val_avg_losses = [np.array(val_losses[i:i+val_iterations]).mean() 
                      for i in range(0, len(val_losses), val_iterations)]
    l1, = plt.plot(epoches_list, train_avg_losses, c="blue")
    l2, = plt.plot(epoches_list, val_avg_losses, c="red")
    plt.legend(handles=[l1, l2], 
               labels=["Train loss", "Validation loss"], loc="best")
    plt.show()

def plot_power_scenarios(gen_scenarios: torch.Tensor,
                         true_scenario: torch.Tensor, 
                         span: list=[], model: str="", source: str="",
                         save: bool=True, save_path: str=""):
    """ Plot a graph with all the scenarios produced by a model
  
    Parameters
    ----------
    - `gen_scenario`: the generated scenarios
    - `true_scenario`: the real scenario
    - `span`: focus interval of the final plot
    - `model`: name of the model that produced the scenarios
    - `source`: energy source of the dataset used for the forecasting
    - `save`: True if to save quality graphs
    - `save_path`: path where to save quality graphs
    """
    # Figure preparation
    fig = plt.figure(figsize=(15, 5))
    stringsave = f"{model}"
    if span != []:
        if span[1] - span[0] == 24:
            spanperiod = " - day 1"
        elif span[1] - span[0] == 24*7:
            spanperiod = " - week 1"
        elif span[1] - span[0] == 24*7*4:
            spanperiod = " - month 1"
        else:
            spanperiod = ""
    else:
        spanperiod = ""
    plt.title(f"{model.upper()} Power Scenarios [{source}]{spanperiod}")
    plt.xlabel("Sampling periods")
    plt.ylabel("Power")
    periods = list(range(len(true_scenario.numpy())))
    periods = periods[span[0]:span[1]] if span != [] else periods
    true_power = list(true_scenario.numpy())
    true_power = true_power[span[0]:span[1]] if span != [] else true_power

    # Scenarios plotting
    handles, labels = [], []
    for s, scen in enumerate(tqdm(gen_scenarios)):
        pred_power = list(scen.cpu().numpy())
        pred_power = pred_power[span[0]:span[1]] if span != [] else pred_power
        l2, = plt.plot(periods, pred_power, c="lightblue")
        if handles == []:
            handles.append(l2)
            labels.append(f"{model} scenarios")
    l1, = plt.plot(periods, true_power, c="blue")
    handles.append(l1)
    labels.append("True power")
    plt.legend(handles=handles, 
                labels=labels, loc="best")

    # Save plots
    if save:
        focus = f"_focus_{span[0]}_{span[1]}" if span != [] else ""
        plt.savefig(os.path.join(
            save_path, f"{stringsave}_scenarios{focus}.png"))
    #plt.show()

def plot_forecast_quality(gen_scenario: torch.Tensor,
                          true_scenario: torch.Tensor, 
                          span: list=[], models: list=[], source: str="",
                          save: bool=True, save_path: str=""):
    """ Plot a comparison graph between forecasts and measurements
  
    Parameters
    ----------
    - `gen_scenario`: the generated scenario to compare
    - `true_scenario`: the real scenario
    - `span`: focus interval of the final plot
    - `models`: names of models that produced the multiple scenarios
    - `source`: energy source of the dataset used for the comparison
    - `save`: True if to save quality graphs
    - `save_path`: path where to save quality graphs
    """
    # Figure preparation
    plt.rcParams.update({'font.size': 13})  # default size 10
    fig = plt.figure(figsize=(15, 5))
    plt.xlabel("Sampling periods")
    plt.ylabel("Power")
    if span != []:
        if span[1] - span[0] == 24:
            spanperiod = " - day 1"
        elif span[1] - span[0] == 24*7:
            spanperiod = " - week 1"
        elif span[1] - span[0] == 24*7*4:
            spanperiod = " - month 1"
    else:
        spanperiod = ""
    periods = list(range(len(true_scenario.numpy())))
    periods = periods[span[0]:span[1]] if span != [] else periods
    true_power = list(true_scenario.numpy())
    true_power = true_power[span[0]:span[1]] if span != [] else true_power
    l1, = plt.plot(periods, true_power, c="navy")
    plt.legend(handles=[l1],
                    labels=["True power"], loc=1)

    stringsave = "random"

    if not isinstance(gen_scenario, list):
        # Case 1 - single model power
        stringsave = f"{models[0]}_"
        plt.title(f"{models[0].upper()} Power Forecasts " + \
                  f"[{source}]{spanperiod}", pad=10)
        pred_power = list(gen_scenario.cpu().numpy())
        pred_power = pred_power[span[0]:span[1]] if span != [] else pred_power
        l2, = plt.plot(periods, pred_power, c="red")
        plt.legend(handles=[l1, l2], 
                    labels=["True power", "Predicted power"], loc=1)
    else:
        # Case 2 - multiple model power
        # TODO: REMOVE THIS (tmp - just for article visualization)
        models[1] = "qgru" if models[1] == "qgru_2" else models[1]
        handles = [l1]
        labels = ["True power"]
        colors = ["firebrick", "forestgreen",
                  "black", "orange", "cyan", "yellow"]
        colors_chosen = colors[:len(gen_scenario)]
        if len(models) == 2:
            stringsave = f"{models[0]}_{models[1]}_"
            plt.title(f"{models[0].upper()} vs {models[1].upper()} " + 
                      f"Power Forecasts [{source} " * \
                      f"(extended)]{spanperiod}", pad=10)
        elif len(models) == 3:
            stringsave = f"{models[0]}_{models[1]}_{models[2]}_"
            plt.title(f"{models[0].upper()} vs {models[1].upper()} " + \
                      f"vs {models[2].upper()} " + \
                      f"Power Forecasts [{source}]{spanperiod}", pad=10)
        for s, scen in enumerate(gen_scenario):
            pred_power = list(scen.cpu().numpy())
            pred_power = \
                pred_power[span[0]:span[1]] if span != [] else pred_power
            # NOTE: TMP - to increase QGRU plot size
            linewidth = 1.0 if s == 0 else 2.5
            l2, = plt.plot(periods, pred_power, c=colors_chosen[s], 
                           alpha=0.7, linewidth=linewidth)
            labels.append(f"{models[s]} power forecast")
            handles.append(l2)
        plt.legend(handles=handles, 
                    labels=labels, loc=1)

    # Save plots
    if save:
        focus = spanperiod.replace(" ", "").replace("-", "_")
        plt.savefig(os.path.join(
            save_path, f"{stringsave}comparison{focus}.png"))
    plt.show()