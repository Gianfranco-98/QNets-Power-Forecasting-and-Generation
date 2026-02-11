# Generic
import os
import json
import time
import shutil
import pickle
import argparse
import pandas as pd
from pprint import pprint
from dill.source import getsource

# Learning
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary

# Local files
from config import settings
from qnets_power.dataset import *
from qnets_power.models.vae import *
from qnets_power.models.gan import *
from qnets_power.models.rnn import *
from qnets_power.plot_tools import *
from qnets_power.metrics import EnergyMetricsComputer
from qnets_power.registry import MODEL_REGISTRY
 

if __name__ == "__main__":

    # Args handling
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str.lower, nargs="+",
                        choices=["vae", "rvae", "qvae", "rqvae",
                                 "gan", "qgan", "rnn", "qrnn"],
                        help="select the model(s)")
    parser.add_argument("--energy-source", type=str,
                        choices=["solar", "wind"],
                        help="select the energy source")
    parser.add_argument("--quaternion-mode", type=str,
                        choices=["reshape", "organize", "timestack"],
                        help="select the quaternion data organization scheme")
    parser.add_argument("--train", action="store_true",
                        help="set if to train the specified model(s)")
    parser.add_argument("--test", action="store_true",
                        help="set if to test the specified model(s)")
    parser.add_argument("--speed-check", action="store_true",
                        help="make a forward speed check for the model(s)")
    parser.add_argument("--save-config", action="store_true",
                        help="save global parameters and net(s) structure")
    parser.add_argument("--num-epochs", type=int,
                        help="desired number of train epochs")
    parser.add_argument('--num-scenarios', type=int,
                        help="# scenarios to generate (always 1 for RNN)")
    parser.add_argument("--des-metric", type=str,
                        choices=["mape", "rmse", "mbe", "pcorr",
                                 "mae", "cosi", "crps", "es", "qs"],
                        help="select the desired metric to choose scenarios")
    parser.set_defaults(
        model=[settings["model"]["default"]],
        energy_source=settings["data"]["source"],
        quaternion_mode=settings["data"]["quaternion_mode"],
        num_epochs=settings["train"]["epochs"],
        num_scenarios=settings["data"]["num_scenarios"],
        des_metric=settings["test"]["des_metric"])
    args = parser.parse_args()

    # -------------------------------------------------------------- #
    # ----------------------- Initialization ----------------------- #
    # -------------------------------------------------------------- #
    print(f"\n----------- Initialization -----------")

    # Arguments checking
    ## Quaternion mode selection
    arg_time = time.time()
    print("\n- Checking args ...", end=" ", flush=True)
    q_models = []
    quaternion_mode = None
    for model_name in args.model:
        if args.energy_source == "solar":
            if "vae" in model_name:
                settings["model"][model_name]["latent_space"] = 40
        if "q" in model_name:
            q_models.append(model_name)
            quaternion_mode = args.quaternion_mode
    if len(q_models) > 0 and len(q_models) != len(args.model):
        raise ValueError("\nYou cannot mix quaternion and normal models")
    quaternion_mode_str = "" if quaternion_mode is None else quaternion_mode
    print(f"Done! ({round(time.time() - arg_time, 1)} s)")

    # Device initialization
    device_time = time.time()
    print("\n- Device init ...", end=" ", flush=True)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = min(1, torch.cuda.device_count())
    if device.type == "cuda":
        accelerator = "gpu"
        devices = n_gpus
    elif device.type == "cpu":
        accelerator = "cpu"
        devices = 1
    torch.set_float32_matmul_precision(
        settings["train"]["matmul_precision"])
    print(f"Done! ({round(time.time() - device_time, 1)} s)")

    # Dataset initialization
    data_time = time.time()
    print("\n- Dataset and Data Module init ...")
    pl.seed_everything(settings["data"]["pl_seed"])
    pd.options.mode.chained_assignment = None
    if args.energy_source == "solar":
        dataset = Solar2014Dataset(split="complete")
    elif args.energy_source == "wind":
        dataset = Wind2014Dataset(split="complete")
    else:
        raise ValueError("Wind and solar are the only available sources")
    pow_hours = 24 - np.array(dataset.zones_zero_pow).shape[-1]
    train_dataset, val_dataset, test_dataset = dataset.split_all()
    datamodule = GEFCom2014DataModule(
        train_dataset, val_dataset, test_dataset,
        batch_size=settings["train"]["batch_size"],
        quaternion_mode=quaternion_mode)
    print(f"Done! ({round(time.time() - data_time, 1)} s)")
    train_dataset.general_info()
    val_dataset.general_info()
    test_dataset.general_info()

    # Model initialization
    models = {}
    for model_name in args.model:
        model_time = time.time()
        print(f"\n{model_name.upper()} Model init ...", end=" ", flush=True)

        # Initialize model-specific paths
        model_checkpoints_path = os.path.join(
            settings["path"]["checkpoints"], model_name,
            args.energy_source, quaternion_mode_str)
        os.makedirs(model_checkpoints_path, exist_ok=True)     

        # Initialize model class and args
        in_cond = datamodule.get_inference_size(phase="train")
        model_config = settings["model"][model_name]
        model_class = MODEL_REGISTRY[model_name]
        model_kwargs = model_class.get_default_kwargs(
            pow_hours, in_cond, device)
        model_kwargs.update(model_config)
        model_kwargs["name"] = model_name
        model = model_class(**model_kwargs).to(device)
        models[model_name] = {
            "model": model,
            "kwargs": model_kwargs,
            "paths": {"checkpoints": model_checkpoints_path}}

        # Show model info
        print(f"Done! ({round(time.time() - model_time, 1)} s)")
        print("Model Parameters =", sum(
            p.numel() for p in model.parameters() if p.requires_grad))
        print("Model Architecture:\n", model)

    # -------------------------------------------------------------- #
    # -------------------------- Training -------------------------- #
    # -------------------------------------------------------------- #
    if args.train:

        # Training setup
        train_setup_time = time.time()
        print(f"\nTraining setup ...", end=" ", flush=True)
        datamodule.setup(stage="fit")
        n_batches = int(len(datamodule.train_dataloader().dataset) // 
                        settings["train"]["batch_size"])
        print(f"Done! ({round(time.time() - train_setup_time, 1)} s)")

        # Training loop
        for model_name, model_dict in models.items():
            train_time = time.time()
            print(f"\n----------- {model_name.upper()} training -----------")

            ## Checkpoint callback specification
            if model_name in ["gan", "qgan"]:
                #checkpoint_monitor = "val_dis_loss"
                checkpoint_monitor = None
                save_top_k = 1
            else:
                checkpoint_monitor = \
                    settings["train"]["checkpoint"]["monitor"]
                save_top_k = settings["train"]["top_k_save"]
            checkpoint_callback = ModelCheckpoint(
                dirpath=model_dict["paths"]["checkpoints"], save_last=True,
                monitor=checkpoint_monitor, save_top_k=save_top_k,
                every_n_epochs=settings["train"]["checkpoint"]["period"])

            ## Trainer initialization
            trainer = pl.Trainer(
                accelerator=accelerator, devices=devices,
                callbacks=[checkpoint_callback],
                max_epochs=args.num_epochs, log_every_n_steps=n_batches/2,
                default_root_dir=settings["path"]["root"])

            ## Model training
            model = model_dict["model"]
            trainer.fit(model, datamodule)

            # Additional checkpoint informations
            with open(os.path.join(
                    model_dict["paths"]["checkpoints"],
                        "checkpoints_info.pt"), "wb") as f:
                pickle.dump(checkpoint_callback.best_k_models, 
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            best_path = os.path.join(
                model_dict["paths"]["checkpoints"], "best")
            os.makedirs(best_path, exist_ok=True)
            shutil.copy(checkpoint_callback.best_model_path, best_path)
            print(f"\n{model_name.upper()} training completed in " + \
                  f"{round(time.time() - train_time, 1)} s.\n")

    # --------------------------------------------------------------- #
    # --------------------------- Testing --------------------------- #
    # --------------------------------------------------------------- #
    want_test = ""
    if not args.test:
        while not want_test in ["y", "n"]:
            want_test = input(
                "Do you want to perform model testing? (y/n):")
    if args.test or want_test.lower() == "y":

        # Testing setup
        test_setup_time = time.time()
        print(f"\nTesting setup ...", end=" ", flush=True)
        datamodule.setup(stage="test")
        metrics_computer = EnergyMetricsComputer(["all"], device=device)
        print(f"Done! ({round(time.time() - test_setup_time, 1)} s)")

        # Testing loop
        for model_name, model_dict in models.items():
            test_time = time.time()
            print(f"\n----------- {model_name.upper()} testing -----------")

            # Initialize model-specific paths
            model_scenarios_path = os.path.join(
                settings["path"]["results"], model_name,
                args.energy_source, quaternion_mode_str, "scenarios")
            model_metrics_path = os.path.join(
                settings["path"]["results"], model_name,
                args.energy_source, quaternion_mode_str, "metrics")
            model_plots_path = os.path.join(
                settings["path"]["results"], model_name,
                args.energy_source, quaternion_mode_str, "plots")
            model_speed_path = os.path.join(
                settings["path"]["results"], model_name,
                args.energy_source, quaternion_mode_str, "speed")
            os.makedirs(model_scenarios_path, exist_ok=True)
            os.makedirs(model_metrics_path, exist_ok=True)
            os.makedirs(model_plots_path, exist_ok=True)
            os.makedirs(model_speed_path, exist_ok=True)
            model_dict["paths"].update({
                "scenarios": model_scenarios_path,
                "metrics": model_metrics_path,
                "plots": model_plots_path,
                "speed": model_speed_path})

            # Model loading from checkpoints
            checkload_time = time.time()
            checkpoint_method = settings["train"]["checkpoint"]["method"]
            print(f"\n- Model checkpoint loading " + \
                  f"[method={checkpoint_method}] ...", end=" ", flush=True)
            if checkpoint_method == "best":
                best_path = os.path.join(
                    model_dict["paths"]["checkpoints"], "best")
                checkpoint_filepath = os.path.join(
                    best_path, os.listdir(best_path)[0])
            elif checkpoint_method == "last":
                with open(os.path.join(
                        model_dict["paths"]["checkpoints"],
                        "checkpoints_info.pt"), "rb") as f:
                    checkpoints_info = pickle.load(f)
                checkpoint_filepath = list(checkpoints_info.keys())[-1]
            else:
                raise ValueError(
                    f"\nCheckpoint method {checkpoint_method} not supported")
            # NOTE: this is needed only to know the last epoch
            checkpoint = torch.load(checkpoint_filepath)
            model_class = MODEL_REGISTRY[model_name]
            model = model_class.load_from_checkpoint(
                    checkpoint_filepath, map_location=None, hparams_file=None,
                    strict=True, **model_dict["kwargs"]).to(device)
            print(f"Done! ({round(time.time() - checkload_time, 1)} s)")

            # Forward speed checking
            if args.speed_check:
                speed_time = time.time()
                print(f"\n- Speed check ...", end=" ", flush=True)
                forecast_list = []
                try:
                    data_split = datamodule.gefcom_train
                except AttributeError as ae:
                    datamodule.setup(stage="fit")
                    data_split = datamodule.gefcom_train
                days = list(range(len(data_split)))
                start_forward = time.time()
                for d in tqdm(days):
                    with torch.no_grad():
                        vars, power = data_split[d]
                        vars = vars.unsqueeze(0).to(device)
                        if model_name not in ["gan", "qgan"]:
                            if model_name not in ["rnn", "qrnn"]:
                                power = power.unsqueeze(0).to(device)
                                fcasts = model(vars, power)
                            else:
                                fcasts = model(vars)
                        else:
                            noise = torch.randn(
                                vars.shape[0], model.latent_space).to(device)
                            fcasts_gen = model(vars, noise, net="gen")
                            fcasts_dis = model(
                                vars, fcasts_gen.unsqueeze(0), net="dis")
                            fcasts = [fcasts_gen, fcasts_dis]
                    forecast_list.append(fcasts)
                total_forward = time.time() - start_forward
                forecast_day_time = total_forward / len(forecast_list)
                forecasts_per_s = 1 / forecast_day_time
                print(f"Done! ({round(time.time() - speed_time, 1)} s)")
                print(f"{dataset.source}_" +
                      f"train total forecasting time = {total_forward}")
                print(f"{dataset.source}_" +
                      f"train mean forecasting time = {forecast_day_time}")
                print(f"{dataset.source}_" +
                      f"train forecasts/s time = {forecasts_per_s}")

            # Generate scenarios if not already done
            num_scenarios = \
                args.num_scenarios if model_name not in ["rnn", "qrnn"] else 1
            scenarios_name = f"scenarios_{dataset.source}_" + \
                f"{model_name}_{num_scenarios}_{'test'}.tar"
            if (args.train or \
                    scenarios_name not in os.listdir(
                        model_dict["paths"]["scenarios"])):
                scenarios_time = time.time()
                print(f"\n- Scenarios generation ...")
                test_scenarios, speed = datamodule.generate_scenarios(
                    model, "test", num_scenarios, save=True,
                    save_path=model_dict["paths"]["scenarios"])
                print(f"Done! ({round(time.time() - scenarios_time, 1)} s)")
                ## Save model generation/inference speed info
                speed_filename = os.path.join(
                    model_dict["paths"]["speed"], "speed.json")
                with open(speed_filename, "w") as f:
                    json.dump(speed, f, indent=2)

            # Load already generated scenarios
            load_scenarios_time = time.time()
            print(f"\n- Scenarios loading ...", end=" ", flush=True)
            test_scenarios = datamodule.load_scenarios(
                model, model_dict["paths"]["scenarios"],
                num_scenarios=num_scenarios, split="test")                    
            print(f"Done! ({round(time.time() - load_scenarios_time, 1)} s)")

            # Scenarios reshaping
            prep_scenarios_time = time.time()
            print(f"\n- Scenarios processing ...", end=" ", flush=True)
            test_scenarios_reshaped = test_scenarios.reshape(
                len(test_scenarios) // num_scenarios, num_scenarios, 24)
            test_scenarios_reshaped = test_scenarios_reshaped.permute(1, 0, 2)
            test_scenarios_reshaped = \
                test_scenarios_reshaped.reshape(num_scenarios, -1)
            true_scenario = torch.from_numpy(
                test_dataset.dataframe["POWER"].values)
            print(f"Done! ({round(time.time() - prep_scenarios_time, 1)} s)")

            # Computing metrics
            metrics_time = time.time()
            print(f"\n- Metrics computation ...", end=" ", flush=True)
            metric_results = metrics_computer(
                test_scenarios_reshaped, true_scenario)
            print(f"Done! ({round(time.time() - metrics_time, 1)} s)")
            print("\nMetrics results:")
            pprint(metric_results)
            metrics_filename = os.path.join(
                model_dict["paths"]["metrics"], "metrics.json")
            with open(metrics_filename, "w") as f:
                json.dump(metric_results, f, indent=2)

            # Best scenario selection
            met_sel = args.des_metric
            energy_metrics = EnergyMetricsComputer.energy_metrics
            if met_sel in energy_metrics["error metrics"]:
                metric_type = "error metrics"
            elif met_sel in energy_metrics["statistical metrics"]:
                metric_type = "statistical metrics"
            elif met_sel in energy_metrics["extra metrics"]:
                metric_type = "extra metrics"
            best_idx = metric_results[metric_type][met_sel]["best_idx"]
            best_scenario = test_scenarios_reshaped[best_idx]

            # Scenarios visualization
            ## 1 - first day scenarios
            plot_power_scenarios(
                test_scenarios_reshaped, true_scenario, span=[0, 24],
                model=model_name, source=args.energy_source,
                save=True, save_path=model_dict["paths"]["plots"])
            ## 2 - first week scenarios
            plot_power_scenarios(
                test_scenarios_reshaped, true_scenario, span=[0, 24*7],
                model=model_name, source=args.energy_source,
                save=True, save_path=model_dict["paths"]["plots"])

            # True vs Pred power comparison
            ## 1 - all data forecast
            plot_forecast_quality(
                best_scenario, true_scenario, 
                models=[model_name], source=args.energy_source,
                save=True, save_path=model_dict["paths"]["plots"])
            ## 2 - first month forecast
            plot_forecast_quality(
                best_scenario, true_scenario, span=[0, 24*7*4],
                models=[model_name], source=args.energy_source,
                save=True, save_path=model_dict["paths"]["plots"])
            ## 3 - first week forecast
            plot_forecast_quality(
                best_scenario, true_scenario, span=[0, 24*7],
                models=[model_name], source=args.energy_source,
                save=True, save_path=model_dict["paths"]["plots"])
            print(f"\n{model_name.upper()} testing completed in " + \
                  f"{round(time.time() - test_time, 1)} s.")

    # -------------------------------------------------------------- #
    # --------------------- Save configuration --------------------- #
    # -------------------------------------------------------------- #
    want_save = ""
    if not args.save_config:
        while not want_save in ["y", "n"]:
            want_save = input("Do you want to save global config? (y/n):")
    if args.save_config or want_save.lower() == "y":
        global_config_dict = {}
        global_config_dict["model"] = {}
        global_config_dict["all_params"] = settings.to_dict()
        global_config_dict["all_params"]["DEF_DEVICE"] = \
            str(global_config_dict["all_params"]["DEF_DEVICE"])

        # Saving loop
        config_time = time.time()
        print(f"\n- Saving configuration ...", end=" ", flush=True)
        for model_name, model_dict in models.items():
            
            # Initialize model-specific paths
            model_config_path = os.path.join(
                settings["path"]["results"], model_name,
                args.energy_source, quaternion_mode_str, "config")
            os.makedirs(model_config_path, exist_ok=True)
            model_dict["paths"]["config"] = model_config_path

            # Save model general config
            model = model_dict["model"]
            model_representation = str(model).split("\n")
            model_loss = getsource(model.loss.forward).split("\n")
            global_config_dict["model"] = {
                "name": model_name.upper(),
                "net_structure": model_representation,
                "net_loss": model_loss}

            # Save model inference info
            summary = ModelSummary(model, max_depth=1)
            model_size = str(summary).split("\n")
            speed_filename = os.path.join(
                model_dict["paths"]["speed"], "speed.json")
            with open(speed_filename, "r") as f:
                ## NOTE: Speed file is created when generaing scenarios
                speed_dict = json.load(f)
            if args.speed_check:
                speed_dict["forward_s"] = forecasts_per_s
            global_config_dict["model"]["size"] = model_size
            global_config_dict["model"]["speed"] = speed_dict

            # Dump configuration into file
            config_filename = os.path.join(
                model_dict["paths"]["config"], "configuration.json")
            with open(config_filename, "w") as f:
                json.dump(global_config_dict, f, indent=2)
        print(f"Done! ({round(time.time() - config_time, 1)} s)")