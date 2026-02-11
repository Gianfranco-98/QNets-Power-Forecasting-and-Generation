# Generic
import os
import torch
from dynaconf import Dynaconf


# Import settings from toml files
settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml', '.secrets.toml'],
)


# Custom path settings
## Base paths
settings["path"]["root"] = os.path.dirname(__file__)
settings["path"]["pkg"] = os.path.join(
    settings["path"]["root"], "src", "qnets_power")
settings["path"]["models"] = os.path.join(
    settings["path"]["pkg"], settings["path"]["models_folder"])
## Dataset paths
settings["path"]["dataroot"] = os.path.join(
    settings["path"]["root"], settings["path"]["data_folder"],
    "sets", settings["data"]["name"])
settings["path"]["solar"] = os.path.join(
    settings["path"]["dataroot"], "Solar")
settings["path"]["wind"] = os.path.join(
    settings["path"]["dataroot"], "Wind")
settings["path"]["wind_power"] = os.path.join(
    settings["path"]["wind"],
    f"Task {settings['data']['task']}",
    f"Task{settings['data']['task']}_" + \
    settings['data']['wind']['zones'])
settings["path"]["wind_expvar"] = os.path.join(
    settings["path"]["wind"],
    f"Task {settings['data']['task']}",
    f"TaskExpVars{settings['data']['task']}_" + \
    settings['data']['wind']['zones'])
## Other paths
settings["path"]["normalizer"] = os.path.join(
    settings["path"]["root"], settings["path"]["normalizer_folder"])
settings["path"]["checkpoints"] = os.path.join(
    settings["path"]["root"], settings["path"]["checkpoints_folder"])
settings["path"]["results"] = os.path.join(
    settings["path"]["root"], settings["path"]["results_folder"])

# Num workers for data loading
if os.name == "nt":
    settings["data"]["num_workers"] = \
        settings["data"]["max_num_workers"]["windows"]
else:
    settings["data"]["num_workers"] = \
        settings["data"]["max_num_workers"]["linux"]
    
# Define global default device
settings["def_device"] = torch.device("cpu")