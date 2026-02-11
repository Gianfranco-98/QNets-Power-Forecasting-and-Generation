# Generic
import os
import time
import pickle
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from abc import abstractmethod
from typing import TypeVar, Tuple, Union, List, Any

# Learning
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

# Local files
from config import settings


# Dataset typing preparation
SolarClass = TypeVar("SolarClass", bound="Solar2014Dataset")
WindClass = TypeVar("WindClass", bound="Wind2014Dataset")


# ------------------------------------------------------------------- #
# ----------------------------- Dataset ----------------------------- #
# ------------------------------------------------------------------- #

class RenewableEnergyDataset(torch.utils.data.Dataset):
    """ Base Class for Renewable Energy Datasets """

    def __init__(self, dataset: Any, name: str, split: str,
                 dataroot: str, source: str, all_vars: bool,
                 inference_mode: bool, num_workers: int):
        """ Dataset Initialization

        Parameters
        ----------
        - `dataset`: the instantiated dataset
        - `name`: name of the dataset
        - `split`: the data split ("complete", "train", "val", "test")
        - `dataroot`: the root directory of the dataset
        - `source`: the renewable energy source
        - `all_vars`: True if to use original + derived vars
        - `inference_mode`: True if the model is in inference mode
        - `num_workers`: num of processes that collect data
        """
        super(RenewableEnergyDataset, self).__init__()
        self.x_data = dataset
        self.y_data = None
        self.name = name + " " + source
        self.split = split
        self.dataroot = dataroot
        self.source = source
        self.all_vars = all_vars
        self.inference_mode = inference_mode
        self.num_workers = num_workers

    @abstractmethod
    def __len__(self):
        """ Return the size of the dataset """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int):
        """ Return an element of the dataset """
        raise NotImplementedError

    @abstractmethod
    def split_all(self):
        """ Return the train, test, and val split of the class """

    @abstractmethod
    def build_for_inference(self, quaternion_mode: Union[str, None]):
        """ Build dataset for inference mode """
        raise NotImplementedError

    def general_info(self):
        print("___________ General Dataset Informations ___________\n")
        print(f"- Dataset name: {self.name} [{self.split}]")
        print(f"- Number of samples: {len(self)}")
        if self.y_data is None:
            print(f"- Data sample:\n {self.x_data[0:5]}")
        else:
            print(f"- X data sample:\n {self.x_data[0:5]}")
            print(f"- y data sample:\n {self.y_data[0:5]}")
        print("____________________________________________________\n")
        

class GEFCom2014Dataset(RenewableEnergyDataset):
    """ GEFCom_2014 Dataset for energy-related challenges """

    def __init__(self, dataset: pd.DataFrame,
                 vars: List[str], split: str="train",
                 dataroot: str=settings["path"]["dataroot"],
                 source: str=settings["data"]["source"],
                 all_vars: bool=True, zones: set={}, zones_zero_pow: list=[],
                 num_workers: int=settings["data"]["num_workers"]):
        """ GEFcom_2014 Dataset initialization

        Parameters
        ----------
        - `dataset`: Pandas DataFrame related to an energy source
        - `vars`: the weather variables related to an energy source
        - `split`: the data split ("complete", "train", "val", "test")
        - `dataroot`: the root directory of the dataset
        - `source`: the energy source (or load) to be analyzed
        - `all_vars`: True if to use original + derived vars
        - `zones`: the zones of the power and weather data collection
        - `zones_zero_pow`: zones with no power produced
        - `num_workers`: num of processes that collect data
        """
        # Base Class initialization
        name = "GEFCom_2014"
        super(GEFCom2014Dataset, self).__init__(
            dataset, name, split, dataroot,
            source, all_vars, False, num_workers)
        self.dataframe = self.x_data

        # Other initializations
        self.zones_zero_pow = zones_zero_pow
        self.vars = vars
        self.zones = zones
        self.max_power = 1.
            
    def __len__(self) -> int:
        """ Return the size of the dataset """
        length = len(self.dataframe)
        if self.inference_mode:
            length = len(self.x_data)
        return length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Return an element of the dataset 

        Parameters
        ----------
        `idx`: index of the element

        Return
        ------
        - `vars`: environment weather variables
        - `power`: power generated at the sample time
        """
        vars, power = None, None
        if not self.inference_mode:
            # Get a sample from the dataframe
            sample = self.dataframe[idx:idx+1]

            # Disassemble the sample
            vars = torch.Tensor(
                [sample[self.vars[i]].item() for 
                i in range(len(self.vars))])
            power = torch.Tensor([sample["POWER"].item()])
        else:
            vars = self.x_data[idx]
            power = self.y_data[idx]

        return vars.float(), power.float()

    def split_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Return the train, test, and val split of the class """
        train_data, test_data, val_data = [], [], []
        for zone in self.zones:
            x = self.dataframe[self.dataframe["ZONEID"] == zone]
            train_splits = [x[spl:spl+24].reset_index(drop=True) 
                            for spl in range(0, len(x), 24)]
            test_splits = [train_splits.pop(
                random.randrange(len(train_splits))) for _ in range(100)]
            val_splits = [test_splits.pop(
                random.randrange(len(test_splits))) for _ in range(50)]
            train_data.append(train_splits)
            test_data.append(test_splits)
            val_data.append(val_splits)
        train_data = pd.concat([df for spl_l in train_data for df in spl_l])
        test_data = pd.concat([df for spl_l in test_data for df in spl_l])
        val_data = pd.concat([df for spl_l in val_data for df in spl_l])
        return train_data, val_data, test_data


class Solar2014Dataset(GEFCom2014Dataset):
    """ GEFCom_2014 Solar Dataset """
    
    def __init__(self, split: str="train",
                 dataframe: Union[pd.DataFrame, None]=None,
                 solar_path: str=settings["path"]["solar"],
                 task_num: int=settings["data"]["task"],
                 less_feats: bool=settings["data"]["solar"]["less_feats"],
                 base_feats: List[str]=settings["data"]["solar"]["base_feats"],
                 solar_vars: List[str]=settings["data"]["solar"]["vars"],
                 acc_fields: List[str]=settings["data"]["solar"]["acc_fields"],
                 acc_r_fields: List[str]=settings["data"]["solar"]["acc_r_fields"],
                 diff_acc: bool=settings["data"]["solar"]["diff_acc"], 
                 past_read_p: int=settings["data"]["solar"]["past_read_p"],
                 num_workers: int=settings["data"]["num_workers"]):
        """ GEFCom_2014 Solar Dataset initialization

        Parameters
        ----------
        - `split`: the data split ("complete", "train", "val", "test")
        - `dataframe`: None if to initialize the dataset from skratch
        - `solar_path`: the root directory of the solar dataset
        - `task_num`: the number of the specific task of the challenge
        - `less_feats`: True if to run with reduced set of features
        - `base_feats`: sufficient set of features for training
        - `solar_vars`: weather variables used in solar prediction
        - `acc_fields`: accumulated fields in the solar variables
        - `acc_r_fields`: solar radiation accumulated fields
        - `diff_acc`: True to differentiate accumulated fields
        - `past_read_p`: #periods between sample value and its reading
        - `num_workers`: num of processes that collect data
        """
        # Dataframe initialization
        prepare_data = True
        if dataframe is None:
            dataframe = pd.read_csv(os.path.join(
                solar_path, f"Task {task_num}", f"predictors{task_num}.csv"))
        elif isinstance(dataframe, pd.DataFrame):
            prepare_data = False
        else:
            raise ValueError(
                "dataframe arg must be None or a Solar Dataframe")
        zones = set(dataframe["ZONEID"])
        self.base_feats = base_feats
        self.zones_zero_pow = []
        self.acc_fields = acc_fields
        self.acc_r_fields = acc_r_fields

        # Base Class initialization
        super(Solar2014Dataset, self).__init__(
            dataframe, solar_vars, split, solar_path,
            "solar", True, zones, self.zones_zero_pow, num_workers)

        # Solar data preparation
        if prepare_data:
            self.prepare_solar_data(
                diff_acc, past_read_p, less_feats)

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """ Return an element of the dataset 
        
        Parameters
        ----------
        `idx`: index of the element

        Return
        ------
        - `vars`: environment weather variables
        - `power`: power generated at the sample time
        - `zone_id`: ID of the zone in which data is collected
        - `timestamp`: data and time at which the sample is collected
        """
        return super().__getitem__(idx)

    def prepare_solar_data(self, diff_acc: bool, 
                           past_read_p: int, less_feats: bool):
        """ Adjust Solar dataframe

        1. If specified, the accumulated fields are differentiated
           in order to recover the original read values.
           With this differentiation we should also convert 
           radiation units from J/m^2 to W/m^2).
        2. Each sample of the solar data refers to the past,
           so we should discard the substitute the dataframe values
           with the ones taken past_read_p periods before 
           (discarding the first day).
           Also, in this step we saves the indices
           of the periods where the photovoltaic power is always 0
        3. Warn if indices collected before are different across zones
        
        Parameters
        ----------
        - `diff_acc`: True to differentiate accumulated fields
        - `past_read_p`: #periods from sample value and reading
        - `less_feats`: True to run with reduced set of features
        """
        # 1 - Accumulated Fields differentiation
        if diff_acc:
            self.differentiate_acc_fields()

        # 2 - Data shifting
        def f(x, self) -> pd.DataFrame:
            ## 2.1 - Shift the data
            timestamps = x["TIMESTAMP"].iloc[24:]
            x = x.shift(periods=past_read_p)[24:]
            x["TIMESTAMP"] = timestamps
            ## 2.2 - Save index of always null power
            max_pow = x["POWER"].values.reshape(-1, 24).max(axis=0)
            self.zones_zero_pow.append(list(np.where(max_pow == 0)[0]))
            return x
        self.dataframe = self.dataframe.groupby(["ZONEID"]).apply(
            f, self, include_groups=True).reset_index(drop=True)

        # 3 - Check that null power indices are equal
        z_counts = [self.zones_zero_pow.count(z) for z in self.zones_zero_pow]
        if len(set(z_counts)) > 1:
            warnings.warn("Power plants have different time zones")
        else:
            self.zones_zero_pow = list(self.zones_zero_pow[0])

        # 4 - Eventually run with a reduced set of features
        if less_feats:
            self.reduce_solar_features()
    
    def differentiate_acc_fields(self):
        """ Differentiate accumulated fields of the Solar dataset """
        for acc_field in self.acc_fields:
            n_days = len(self.dataframe) // 24
            acc_field_values = self.dataframe[acc_field].values
            daily_acc_field = acc_field_values.reshape(n_days, 24)
            if acc_field in self.acc_r_fields:
                daily_acc_field /= 3600
            diff_field = np.diff(daily_acc_field, prepend=0)
            diff_field = np.where(diff_field < 0, 0, diff_field)
            self.dataframe[acc_field] = diff_field.reshape(-1)

    def reduce_solar_features(self):
        """ Remove features from the dataset and make a reduced set:
            1. Surface pressure (VAR134)
            2. Relative humidity at 1000 mbar (VAR157)
            3. sqrt(10-metre U wind component (VAR165) ^ 2 + 
                    10-metre V wind component (VAR166) ^ 2)
            4. Surface solar rad down (VAR169) 
               * 2-metre temperature (VAR167)
        """
        # Derive two new features
        self.dataframe["VAR229"] = (
            self.dataframe["VAR165"] ** 2
            + self.dataframe["VAR166"] ** 2) ** (1/2)
        self.dataframe["VAR230"] = \
            self.dataframe["VAR169"] * self.dataframe["VAR167"]

        # Remove unnecessary features
        new_feats = ["VAR134", "VAR157", "VAR229", "VAR230"]
        for var in self.vars:
            if var not in new_feats:
                self.dataframe.drop(var, inplace=True, axis=1)

        # vars update
        self.dataframe = self.dataframe[[
            c for c in self.dataframe if c not in ["POWER"]] + ["POWER"]]
        self.vars = new_feats

    def split_all(self) -> Tuple[SolarClass, SolarClass, SolarClass]:
        """ Return the train, test, and val split of the class """
        train_df, test_df, val_df = super().split_all()
        train_dataset = Solar2014Dataset("train", 
            train_df, solar_vars=self.vars)
        test_dataset = Solar2014Dataset("test", 
            test_df, solar_vars=self.vars)
        val_dataset = Solar2014Dataset("val", 
            val_df, solar_vars=self.vars)
        train_dataset.zones_zero_pow = self.zones_zero_pow
        test_dataset.zones_zero_pow = self.zones_zero_pow
        val_dataset.zones_zero_pow = self.zones_zero_pow
        return train_dataset, val_dataset, test_dataset

    def build_for_inference(self, quaternion_mode: Union[str, None]=None):
        """ Build dataset for inference mode 

        1. In order to make the dataset suitable for inference,
           it's necessary to reshape the data in order to keep
           power information over all the day in a single row.
           For this reason, timestamp is eliminated and zones
           are reported in a 1-hot encoding manner.
           Then, periods with power always 0 (dark hours) are removed.
        2. All the new variables and zones are concatenated
           and put into the self.x_data variable,
           just to keep the original dataframe structure.
        3. Organize data in quaternions if required, as follows:
              - quaternion_mode = QUATERNION_MODE_RESHAPE
                 -> data is just padded and reshaped
              - quaternion_mode = QUATERNION_MODE_ORGANIZE
                 -> data is organized in a precious way
                    [1]: Humidity-related variables + zones
                    [2]: Pressure-related variables + zones
                    [3]: Wind-related variables + zones
                    [4]: Sun-related variables + zones
              - quaternion_mode = QUATERNION_MODE_TIMESTACK
                 -> day data is divided in 4 time blocks and stacked.
        """
        # 1.1 - Data reshaping and dark hours removal
        var_t = torch.Tensor(self.dataframe[self.vars].values).T
        var_t_days = var_t.view(var_t.shape[0], -1, 24).permute(1, 0, 2)
        if len(np.array(self.zones_zero_pow).shape) > 1:
            # Plants in different time zones
            # NOTE: the number of light hours should be the same
            # TODO: check correctness
            no_zero_pow = [[h for h in list(range(24)) if 
                           h not in self.zones_zero_pow[z]] for
                           z in self.zones]
            var_t_days = var_t_days.view(
                len(self.zones), -1, var_t.shape[1], var_t.shape[2])
            var_t_days = torch.stack(
                [var_t_days[z][:, :, no_zero_pow[z]] for z in self.zones])
            var_t_days = var_t_days.view(-1, var_t.shape[2], var_t.shape[3])
            self.y_data = torch.Tensor(
                self.dataframe["POWER"].values).view(len(self.zones), -1, 24)
            self.y_data = torch.stack(
                [self.y_data[z][:, no_zero_pow[z]] for z in self.zones])
            self.y_data = self.y_data.view(
                len(self.zones) * self.y_data.shape[1], -1)
        else:
            # Plants with the same time zones
            no_zero_pow = [z for z in list(range(24)) if 
                           z not in self.zones_zero_pow]
            var_t_days = var_t_days[:, :, no_zero_pow]
            self.y_data = torch.Tensor(
                self.dataframe["POWER"].values).view(-1, 24)
            self.y_data = self.y_data[:, no_zero_pow]
        var_t_final = var_t_days.reshape(var_t_days.shape[0], -1)

        # 1.2 - one-hot encoding
        zone_t = torch.Tensor(self.dataframe["ZONEID"].values).to(torch.int64)
        zone_t_1_h = F.one_hot(zone_t - 1, num_classes=int(zone_t[-1].item()))
        zone_t_1_h = zone_t_1_h.reshape(-1, 24, len(self.zones))

        # 2 - Concatenation and data storage
        self.x_data = torch.cat([var_t_final, zone_t_1_h[..., 0, :]], axis=1)

        # 3 - Quaternion transformation
        if quaternion_mode is not None:
            # TODO: find the best way to rearrange into quaternion
            # 1° attempt [naive] - only reshaping
            if quaternion_mode == "reshape":
                pad_size = 4 - (self.x_data.shape[-1] % 4)
                self.x_data = F.pad(self.x_data, (0, pad_size))
                self.x_data = self.x_data.reshape(self.x_data.shape[0], 4, -1)
                pad_size = 4 - (self.x_data.shape[-1] % 4)
                self.x_data = F.pad(self.x_data, (0, pad_size))
            # 2° attempt - organize data in a precious way
            elif quaternion_mode == "organize":
                if len(self.vars) > 4:
                    # TODO: find a better way to pad variables
                    zones_squeezed = zone_t_1_h[..., 0, :].squeeze()
                    ## [1] - Humidity-related variables
                    humidity_vars_tmp = torch.stack([
                        var_t_days[..., self.vars.index("VAR78"), :],
                        var_t_days[..., self.vars.index("VAR79"), :],
                        var_t_days[..., self.vars.index("VAR157"), :],
                        var_t_days[..., self.vars.index("VAR164"), :],
                        var_t_days[..., self.vars.index("VAR228"), :]],
                        dim=1)
                    humidity_vars = torch.cat([
                        humidity_vars_tmp.flatten(1), zones_squeezed], -1)
                    ## [2] - Pressure-related variables
                    pressure_vars_tmp = torch.zeros_like(humidity_vars_tmp)
                    pressure_vars_tmp[..., 0, :] = \
                        var_t_days[..., self.vars.index("VAR134"), :]
                    pressure_vars = torch.cat([
                        pressure_vars_tmp.flatten(1), zones_squeezed], -1)
                    ## [3] - Wind-related variables
                    wind_vars_tmp = torch.zeros_like(humidity_vars_tmp)
                    wind_vars_tmp[..., 0:2, :] = torch.stack([
                        var_t_days[..., self.vars.index("VAR165"), :],
                        var_t_days[..., self.vars.index("VAR166"), :]],
                        dim=1)
                    wind_vars = torch.cat([
                        wind_vars_tmp.flatten(1), zones_squeezed], -1)
                    ## [4] - Sun-related variables
                    sun_vars_tmp = torch.zeros_like(humidity_vars_tmp)
                    sun_vars_tmp[..., 0:4, :] = torch.stack([
                        var_t_days[..., self.vars.index("VAR167"), :],
                        var_t_days[..., self.vars.index("VAR169"), :],
                        var_t_days[..., self.vars.index("VAR175"), :],
                        var_t_days[..., self.vars.index("VAR178"), :]],
                        dim=1)
                    sun_vars = torch.cat([
                        sun_vars_tmp.flatten(1), zones_squeezed], -1)
                    ## -> Build the quaternion
                    self.x_data = torch.stack([
                        humidity_vars,
                        pressure_vars,
                        wind_vars, 
                        sun_vars], dim=1)
                    self.x_data = F.pad(self.x_data, (0, 1))
                elif len(self.vars) == 4:
                    zone_t = zone_t_1_h[..., 0, :].unsqueeze(1)
                    zone_t = zone_t.repeat(1, 4, 1)
                    self.x_data = torch.cat([
                        var_t_days.flatten(2), zone_t], axis=2)
                    self.x_data = F.pad(self.x_data, (0, 1))
            # 3° attempt - organize data in 4 time steps
            elif quaternion_mode == "timestack":
                # TODO: adapt for data not multiple of 4
                day_split = var_t_final.shape[1] // 4
                var_t_days = var_t_days.reshape(
                    var_t_days.shape[0], var_t_days.shape[1], 4, -1)
                var_t_days = var_t_days.permute(0, 2, 1, 3)
                zone_t = zone_t_1_h[..., 0, :].unsqueeze(1)
                zone_t = zone_t.repeat(1, 4, 1)
                self.x_data = torch.cat([
                    var_t_days.flatten(2), zone_t], axis=2)
                self.x_data = F.pad(self.x_data, (0, 1))

        self.inference_mode = True


class Wind2014Dataset(GEFCom2014Dataset):
    """ GEFCom_2014 Wind Dataset """

    def __init__(self, split: str="train",
                 dataframe: Union[pd.DataFrame, None]=None,
                 wind_path: str=settings["path"]["wind"],
                 task_num: int=settings["data"]["task"],
                 less_feats: bool=settings["data"]["wind"]["less_feats"],
                 task_zones_path: str=settings["path"]["wind_power"],
                 expvar_path: str=settings["path"]["wind_expvar"],
                 wind_vars: List[str]=settings["data"]["wind"]["vars"],
                 keep_original: bool=settings["data"]["wind"]["keep_original"],
                 num_workers: int=settings["data"]["num_workers"]):
        """ GEFCom_2014 Solar Dataset initialization

        Parameters
        ----------
        - `split`: the data split ("complete", "train", "val", "test")
        - `dataframe`: None if to initialize the dataset from skratch
        - `wind_path`: the root directory of the wind dataset
        - `task_num`: the number of the specific task of the challenge
        - `less_feats`: True if to run with reduced set of features
        - `task_zones_path`: task power path by zones
        - `expvar_path`: task expvar path (part of explanatory vars)
        - `wind_vars`: weather variables used for wind prediction
        - `keep_original`: True to keep original variables in data
        - `num_workers`: num of processes that collect data
        """
        # Dataframe initialization
        prepare_data = True
        if dataframe is None:
            dataframe = pd.read_csv(os.path.join(
                task_zones_path, f"Task{task_num}_W_Zone1.csv"))
        elif isinstance(dataframe, pd.DataFrame):
            prepare_data = False
        else:
            raise ValueError("dataframe arg must be None or a Wind Dataframe")
        zones = set(range(1, len(os.listdir(task_zones_path))+1))
        self.keep_original = keep_original

        # Base Class initialization
        super(Wind2014Dataset, self).__init__(
            dataframe, wind_vars, split, wind_path, 
            "wind", True, zones, [], num_workers)
        
        # Wind data preparation
        if prepare_data:
            self.prepare_wind_data(
                task_num, task_zones_path, expvar_path, less_feats)

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """ Return an element of the dataset 
        
        Parameters
        ----------
        `idx`: index of the element

        Return
        ------
        - `vars`: environment weather variables
        - `power`: power generated at the sample time
        - `zone_id`: ID of the zone in which data is collected
        - `timestamp`: data and time at which the sample is collected
        """
        return super().__getitem__(idx)

    def prepare_wind_data(self, task_num: int, task_zones_dir: str, 
                          expvar_dir: str, less_feats: bool):
        """ Adjust Wind dataframe

        1. Load dataframe with solutions of the expvars unknown power 
        2. The wind zone data is contained in different files, so
           the first thing to do is to load all the zone dataframes;
           there is a missing part for each dataframe zone, 
           that should be added to the previous part, and merged 
           with power informations
        3. Change "TARGETVAR" into "POWER" and set it as the last field
        4. If specified, convert the wind U (zonal) and V (meridional) 
           components into speed, energy and wind direction
        5. Remove NaN values from the dataframe
        
        Parameters
        ----------
        - `task_num`: the number of the involved task
        - `task_zones_dir`: dir with all zones files of a certain task
        - `expvar_dir`: continue of the previous, without power info
        - `less_feats`: True if to run with reduced set of features
        """
        # 1 - Solutions Dataframe loading
        expvar_solutions = pd.read_csv(
            os.path.join(self.dataroot,
                         f"Solution to Task {task_num}", 
                         f"solution{task_num}_W.csv"))

        # 2 - Load reamining dataframes and adjust data
        for zone in self.zones:
            expvar_dataframe = pd.read_csv(
                os.path.join(expvar_dir, 
                             f"TaskExpVars{task_num}_W_Zone{zone}.csv"))
            expvar_dataframe["TARGETVAR"] = list(expvar_solutions[
                expvar_solutions["ZONEID"] == zone]["TARGETVAR"])
            if zone == 1:
                self.dataframe = pd.concat([self.dataframe, expvar_dataframe])
            else:
                zone_dataframe = pd.read_csv(
                    os.path.join(task_zones_dir, 
                                 f"Task{task_num}_W_Zone{zone}.csv"))
                self.dataframe = pd.concat(
                    [self.dataframe, zone_dataframe, expvar_dataframe])

        # 3 - Rename power field and place it to the right
        self.dataframe = self.dataframe.reset_index(drop=True)
        self.dataframe = self.dataframe.rename(columns={"TARGETVAR": "POWER"})
        self.dataframe = self.dataframe[[
            c for c in self.dataframe if c not in ["POWER"]] + ["POWER"]]
        
        # 4 - Derive new speed components from the original u and v
        if not less_feats:
            self.derive_uv_components()

        # 5 - Remove NaN values
        # for i, p_val in self.dataframe["POWER"].items():
        #     if p_val != p_val:
        #         self.dataframe.at[i, "POWER"] = \
        #             self.dataframe.iloc[i-1]["POWER"]
        self.dataframe["POWER"] = self.dataframe["POWER"].ffill()


    def derive_uv_components(self, d: int=1.):
        """ Zonal and Meridional -> speed, energy and direction

        - Wind Speed = sqrt[u^2  + v^2]
        - Wind Energy = 0.5 × d × ws^ 3
        - Wind Direction = 180/π × arctan(u, v)

        Parameters
        ----------
        `d`: density (usually set to 1.0)
        """
        # Conversion U,V -> S,E,D
        wind_speed_10 = np.sqrt(
            self.dataframe["U10"]**2 + self.dataframe["V10"]**2)
        wind_speed_100 = np.sqrt(
            self.dataframe["U100"]**2 + self.dataframe["V100"]**2)
        wind_energy_10 = 0.5 * d *  wind_speed_10**3
        wind_energy_100 = 0.5 * d *  wind_speed_100**3
        wind_direction_10 = (180 / np.pi) * np.arctan2(
            self.dataframe["U10"], self.dataframe["V10"])
        wind_direction_100 = (180 / np.pi) * np.arctan2(
            self.dataframe["U100"], self.dataframe["V100"])

        # Dataframe insertion
        start_idx = len(self.vars) + 2
        self.dataframe.insert(start_idx, "WS10", wind_speed_10)
        self.dataframe.insert(start_idx + 1, "WS100", wind_speed_10)
        self.dataframe.insert(start_idx + 2, "WE10", wind_energy_10)
        self.dataframe.insert(start_idx + 3, "WE100", wind_energy_100)
        self.dataframe.insert(start_idx + 4, "WD10", wind_direction_10)
        self.dataframe.insert(start_idx + 5, "WD100", wind_direction_100)

        # Eventually drop the original variables
        if not self.keep_original:
            for var in self.vars:
                self.dataframe.drop(var, inplace=True, axis=1)
            self.vars = []

        # Variables update
        self.vars += ["WS10", "WS100", "WE10", "WE100", "WD10", "WD100"]

    def split_all(self) -> Tuple[WindClass, WindClass, WindClass]:
        """ Return the train, test, and val split of the class """
        train_df, test_df, val_df = super().split_all()
        train_dataset = Wind2014Dataset("train", 
            train_df, wind_vars=self.vars)
        test_dataset = Wind2014Dataset("test", 
            test_df, wind_vars=self.vars)
        val_dataset = Wind2014Dataset("val", 
            val_df, wind_vars=self.vars)
        return train_dataset, val_dataset, test_dataset

    def build_for_inference(self, quaternion_mode: Union[str, None]=None):
        """ Build dataset for inference mode 

        1. In order to make the dataset suitable for inference,
           it's necessary to reshape the data in order to keep
           power informations over all the day in a single row.
           For this reason, timestamp is eliminated and zones
           are reported in a 1-hot encoding manner.
        2. All the new data is concatenated and put into the 
           self.x_data and self.y_data variables, in order to keep
           the original dataframe structure
        3. Organize data in quaternions if required, as follows:
           these possible schemes:
              - quaternion_mode = QUATERNION_MODE_RESHAPE
                 -> data is just padded and reshaped
              - quaternion_mode = QUATERNION_MODE_ORGANIZE
                 -> data is organized in a precious way:
                    - When num_features > 4:
                        [1]: original variables u and v
                        [2]: derived speed variables
                        [3]: derived energy variables
                        [4]: derived direction variables
                    - When num_features == 4:
                        [1]: feature1 + zones
                        [2]: feature2 + zones
                        [3]: feature3 + zones
                        [4]: feature4 + zones
              - quaternion_mode = QUATERNION_MODE_TIMESTACK
                 -> day data is divided in 4 time blocks and stacked.
        """
        # 1 - Data reshaping and one-hot encoding
        var_t = torch.Tensor(self.dataframe[self.vars].values).T
        var_t_days = var_t.view(var_t.shape[0], -1, 24).permute(1, 0, 2)
        var_t_final = var_t_days.reshape(var_t_days.shape[0], -1)
        zone_t = torch.Tensor(self.dataframe["ZONEID"].values).to(torch.int64)
        zone_t_1_h = F.one_hot(zone_t - 1, num_classes=int(zone_t[-1].item()))
        zone_t_1_h = zone_t_1_h.reshape(-1, 24, len(self.zones))

        # 2 - Concatenation and data storage
        self.x_data = torch.cat([var_t_final, zone_t_1_h[..., 0, :]], axis=1)
        self.y_data = torch.Tensor(
            self.dataframe["POWER"].values).view(-1, 24)

        # 3 - Quaternion transformation
        if quaternion_mode is not None:
            # TODO: find the best way to rearrange into quaternion
            # 1° attempt [naive] - only reshaping
            if quaternion_mode == "reshape":
                pad_size = 4 - (self.x_data.shape[-1] % 4)
                self.x_data = F.pad(self.x_data, (0, pad_size))
                self.x_data = self.x_data.reshape(self.x_data.shape[0], 4, -1)
                pad_size = 4 - (self.x_data.shape[-1] % 4)
                self.x_data = F.pad(self.x_data, (0, pad_size))
            # 2° attempt - organize data in a certain way
            elif quaternion_mode == "organize":
                if len(self.vars) > 4:
                    zones_squeezed = zone_t_1_h[..., 0, :].squeeze()
                    ## [1] - Original
                    original_vars_tmp = torch.stack([
                        var_t_days[:, self.vars.index("U10"), :],
                        var_t_days[:, self.vars.index("V10"), :],
                        var_t_days[:, self.vars.index("U100"), :],
                        var_t_days[:, self.vars.index("V100"), :]], 1)
                    original_vars = torch.cat([
                        original_vars_tmp.flatten(1), zones_squeezed], -1)
                    ## [2] - Speed variables
                    speed_vars_tmp = torch.zeros_like(original_vars_tmp)
                    speed_vars_tmp[..., 0:2, :] = torch.stack([
                        var_t_days[:, self.vars.index("WS10"), :],
                        var_t_days[:, self.vars.index("WS100"), :]], 1)
                    speed_vars = torch.cat([
                        speed_vars_tmp.flatten(1), zones_squeezed], -1)
                    ## [3] - Energy variables
                    energy_vars_tmp = torch.zeros_like(original_vars_tmp)
                    energy_vars_tmp[..., 0:2, :] = torch.stack([
                        var_t_days[:, self.vars.index("WE10"), :],
                        var_t_days[:, self.vars.index("WE100"), :]], 1)
                    energy_vars = torch.cat([
                        energy_vars_tmp.flatten(1), zones_squeezed], -1)
                    ## [4] - Direction variables
                    direction_vars_tmp = torch.zeros_like(original_vars_tmp)
                    direction_vars_tmp[..., 0:2, :] = torch.stack([
                        var_t_days[:, self.vars.index("WD10"), :],
                        var_t_days[:, self.vars.index("WD100"), :]], 1)
                    direction_vars = torch.cat([
                        direction_vars_tmp.flatten(1), zones_squeezed], -1)
                    ## -> Build the quaternion
                    # TODO: implement for different number of 10 and 100 vars
                    self.x_data = torch.stack(
                        [original_vars, 
                         speed_vars, 
                         energy_vars, 
                         direction_vars], dim=1)
                    self.x_data = F.pad(self.x_data, (0, 2))
                elif len(self.vars) == 4:
                    zone_t = zone_t_1_h[..., 0, :].unsqueeze(1)
                    zone_t = zone_t.repeat(1, 4, 1)
                    self.x_data = torch.cat([
                        var_t_days.flatten(2), zone_t], axis=2)
                    self.x_data = F.pad(self.x_data, (0, 2))
            # 3° attempt - organize data in 4 time steps
            elif quaternion_mode == "timestack":
                # TODO: adapt for data not multiple of 4
                day_split = var_t_final.shape[1] // 4
                var_t_days = var_t_days.reshape(
                    var_t_days.shape[0], var_t_days.shape[1], 4, -1)
                var_t_days = var_t_days.permute(0, 2, 1, 3)
                zone_t = zone_t_1_h[..., 0, :].unsqueeze(1)
                zone_t = zone_t.repeat(1, 4, 1)
                self.x_data = torch.cat([
                    var_t_days.flatten(2), zone_t], axis=2)
                self.x_data = F.pad(self.x_data, (0, 2))

        self.inference_mode = True


# -------------------------------------------------------------------------- #
# ------------------------------ Data Modules ------------------------------ #
# -------------------------------------------------------------------------- #

class RenewableEnergyDataModule(pl.LightningDataModule):
    """ PyTorch Lightning Data Module for Renewable Energy """

    def __init__(self,
                 train_dataset: RenewableEnergyDataset,
                 val_dataset: RenewableEnergyDataset,
                 test_dataset: Union[RenewableEnergyDataset, None]=None,
                 quaternion_mode: Union[str, None]=None,
                 batch_size: int=settings["train"]["batch_size"],
                 normalization: str=settings["data"]["normalization"],
                 normalizer_dir: str=settings["path"]["normalizer"],
                 num_workers: int=settings["data"]["num_workers"]):
        """ Data Module initialization

        Parameters
        ----------
        - `train_dataset`: instance of the train dataset class
        - `val_dataset`: instance of the validation dataset class
        - `test_dataset`: instance of the test dataset class
        - `quaternion_mode`: specifies the quaternion data organization
        - `batch_size`: #samples/iter to extract from the dataset
        - `normalization`: which kind of data normalization to apply
        - `normalizer_dir`: directory where to save the data normalizer
        - `num_workers`: number of cores implied in data collection
        """
        super(RenewableEnergyDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        if test_dataset is None:
            self.test_dataset = val_dataset
        self.all_vars = train_dataset.all_vars
        self.quaternion_mode = quaternion_mode
        self.batch_size = batch_size
        self.normalization = normalization
        self.x_normalizer = None
        self.y_normalizer = None
        self.normalizer_dir = normalizer_dir
        self.num_workers = num_workers
        self.setup_fit_done = False
        self.setup_test_done = False

    def setup(self, stage: Union[str, None]=None):
        """ Setup the data module """
        normalize_data = False
        if (stage == "fit" or stage is None) and not self.setup_fit_done:
            self.train_dataset.build_for_inference(self.quaternion_mode)
            self.val_dataset.build_for_inference(self.quaternion_mode)
            self.setup_fit_done = True
            normalize_data = True
        if (stage == "test" or stage is None) and not self.setup_test_done:
            self.test_dataset.build_for_inference(self.quaternion_mode)
            self.setup_test_done = True
            normalize_data = True
        if self.normalization is not None and normalize_data:
            self.normalize_data(stage)

    def normalize_data(self, stage: Union[str, None]=None):
        """ Normalize data in order to avoid unbalancing """
        # Normalizer location on disk
        quat_str = ""
        source_str = self.train_dataset.source
        vars_str = "all_vars" if self.all_vars else "der_vars"
        if self.quaternion_mode == "reshape":
            quat_str = "_quat_resh"
        if self.quaternion_mode == "organize":
            quat_str = "_quat_org"
        if self.quaternion_mode == "timestack":
            quat_str = "_quat_time"
        x_norm_name = f"x_normalizer_{source_str}_{vars_str + quat_str}.pt"
        y_norm_name = f"y_normalizer_{source_str}_{vars_str + quat_str}.pt"
        x_norm_filename = os.path.join(
            self.normalizer_dir, x_norm_name)
        y_norm_filename = os.path.join(
            self.normalizer_dir, y_norm_name)

        # Normalization
        if stage == "fit":
            # Flatten data for quaternions
            if self.quaternion_mode is not None:
                quat_1_shape = self.train_dataset.x_data.shape[1]
                self.train_dataset.x_data = \
                    self.train_dataset.x_data.flatten(1)
                self.val_dataset.x_data = \
                    self.val_dataset.x_data.flatten(1)
            # Fit the normalizer
            if self.x_normalizer is None:
                # Standardization
                if self.normalization == "std_scale":
                    #norm_filenames = \
                    #    [f for f in os.listdir(self.normalizer_dir) if 
                    #     source_str + "_" + vars_str + quat_str in f]
                    #if len(norm_filenames) == 0:
                    if x_norm_name not in os.listdir(self.normalizer_dir):
                        # Fit std scaler
                        self.x_normalizer = StandardScaler()
                        self.y_normalizer = StandardScaler()
                        self.x_normalizer.fit(self.train_dataset.x_data)
                        self.y_normalizer.fit(self.train_dataset.y_data)

                        # Save it to disk
                        with open(x_norm_filename, "wb") as f:
                            pickle.dump(self.x_normalizer, f)
                        with open(y_norm_filename, "wb") as f:
                            pickle.dump(self.y_normalizer, f)
                    else:
                        # Load normalizer from disk
                        with open(x_norm_filename, "rb") as f:
                            self.x_normalizer = pickle.load(f)
                        with open(y_norm_filename, "rb") as f:
                            self.y_normalizer = pickle.load(f)

            # Transform train and validation data
            self.train_dataset.x_data = torch.from_numpy(
                self.x_normalizer.transform(self.train_dataset.x_data))
            self.train_dataset.y_data = torch.from_numpy(
                self.y_normalizer.transform(self.train_dataset.y_data))
            self.val_dataset.x_data = torch.from_numpy(
                self.x_normalizer.transform(self.val_dataset.x_data))
            self.val_dataset.y_data = torch.from_numpy(
                self.y_normalizer.transform(self.val_dataset.y_data))

            # Restore data for quaternions
            if self.quaternion_mode is not None:
                self.train_dataset.x_data = \
                    self.train_dataset.x_data.reshape(
                        self.train_dataset.x_data.shape[0], quat_1_shape, -1)
                self.val_dataset.x_data = \
                    self.val_dataset.x_data.reshape(
                        self.val_dataset.x_data.shape[0], quat_1_shape, -1)

        elif stage == "test":
            # Flatten data for quaternions
            if self.quaternion_mode is not None:
                self.test_dataset.x_data = \
                    self.test_dataset.x_data.flatten(1)
            # Load normalizer
            if self.x_normalizer is None:
                if len(os.listdir(self.normalizer_dir)) == 0:
                    raise ValueError(
                        "Normalizer cannot be None during test phase")
                else:
                    # Load normalizer from disk
                    with open(x_norm_filename, "rb") as f:
                        self.x_normalizer = pickle.load(f)
                    with open(y_norm_filename, "rb") as f:
                        self.y_normalizer = pickle.load(f)
            
            # Transform test data
            self.test_dataset.x_data = torch.from_numpy(
                self.x_normalizer.transform(self.test_dataset.x_data))
            self.test_dataset.y_data = torch.from_numpy(
                self.y_normalizer.transform(self.test_dataset.y_data))

            # Restore data for quaternions
            if self.quaternion_mode is not None:
                self.test_dataset.x_data = \
                    self.test_dataset.x_data.reshape(
                        self.test_dataset.x_data.shape[0], 4, -1)

    def get_inference_size(self, phase: str="train") -> int:
        """ Get length of inference-prepared data

        Parameters
        ----------
        `phase`: train | val | test, depending on which data to infer

        Return
        ------
        `inf_size`: length of inference-prepared data
        """
        # Dataset selection
        if phase == "train":
            dataset = self.train_dataset
        elif phase == "val":
            dataset = self.val_dataset
        elif phase == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown {phase} phase")

        # Temporary data shape extraction
        tmp_dataset = deepcopy(dataset)
        tmp_dataset.build_for_inference(self.quaternion_mode)
        return tmp_dataset.x_data.shape[-1]

    @abstractmethod
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """ Dataloader for the training part """
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """ Dataloader for the validation part """
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """ Dataloader for the testing part """
        raise NotImplementedError


class GEFCom2014DataModule(RenewableEnergyDataModule):
    """ PyTorch Lightning Data Module for Renewable Energy """

    def __init__(self,
                 gefcom_train: GEFCom2014Dataset,
                 gefcom_val: GEFCom2014Dataset,
                 gefcom_test: GEFCom2014Dataset,
                 quaternion_mode: Union[str, None],
                 batch_size: int=settings["train"]["batch_size"],
                 normalization: str=settings["data"]["normalization"],
                 normalizer_dir: str=settings["path"]["normalizer"],
                 num_workers: int=settings["data"]["num_workers"]):
        """ Data Module initialization

        Parameters
        ----------
        - `gefcom_train`: GEFCom2014 class instance (split="train")
        - `gefcom_val`: GEFCom2014 class instance (split="val")
        - `gefcom_test`: GEFCom2014 class instance (split="test")
        - `quaternion_mode`: specifies the quaternion data organization
        - `batch_size`: #samples/iter to extract from the dataset
        - `normalization`: which kind of data normalization to apply
        - `normalizer_dir`: directory where to save the data normalizer
        - `num_workers`: number of cores implied in data collection
        """
        super(GEFCom2014DataModule, self).__init__(
            gefcom_train, gefcom_val, gefcom_test, quaternion_mode,
            batch_size, normalization, normalizer_dir, num_workers)

    def setup(self, stage: Union[str, None]=None):
        """ Setup the data module """
        super().setup(stage)
        if (stage == "fit" or stage is None):
            self.gefcom_train = self.train_dataset
            self.gefcom_val = self.val_dataset
        if (stage == "test" or stage is None):
            self.gefcom_test = self.test_dataset

    def generate_scenarios(self, model: pl.LightningModule, split: str="train",
                           num_scenarios: int=settings["data"]["num_scenarios"],
                           days: list=[], save: bool=False, save_path: str=""
                           ) -> Tuple[torch.Tensor, dict]:
        """ Generate power scenarios with the given model 
        
        Parameters
        ----------
        - `model`: the (trained) generative model to sample scenarios
        - `split`: the dataset split from which to make the generation
        - `num_scenarios`: # scenarios to produce (per day)
        - `days`: if specified, list with idcs of days to generate
        - `save`: True if to save scenarios on disk
        - `save_path`: path where to save generated scenarios

        Return
        ------
        - `generated_scenarios`: generated power scenarios
        - `scenarios_speed`: model scenario generation speed dict
        """
        # Generation setup
        if split == "train":
            try:
                dataset = self.gefcom_train
            except AttributeError as ae:
                self.setup(stage="fit")
                dataset = self.gefcom_train
        elif split == "val":
            dataset = self.gefcom_val
        elif split == "test":
            dataset = self.gefcom_test
        generated_scenarios = []
        days = list(range(len(dataset))) if days == [] else days

        # Scenarios generation
        start_gen = time.time()
        for d in tqdm(days):
            vars, power = dataset[d]
            vars = vars.to(model.device)
            scenarios = model.generate(vars, num_scenarios)
            scenarios = torch.from_numpy(
                self.y_normalizer.inverse_transform(scenarios.cpu()))
            scenarios = torch.clamp(scenarios, min=0., max=dataset.max_power)
            if dataset.source == "solar":
                # Re-fill dark hours with 0 power values
                # TODO: implement also the multi time zones case
                complete_scenarios = torch.zeros(num_scenarios, 24).double()
                no_zero_pow = [z for z in list(range(24)) if 
                               z not in dataset.zones_zero_pow]
                complete_scenarios[..., no_zero_pow] = scenarios
                scenarios = complete_scenarios
            generated_scenarios.append(scenarios)
        total_gen = time.time() - start_gen
        scen_day_time = total_gen / len(generated_scenarios)
        scen_per_s = 1 / (100 * scen_day_time)
        print(f"{dataset.source}_{split}" +
              f"100 at day total generation time = {total_gen}")
        print(f"{dataset.source}_{split}" +
              f"100 at day mean generation time = {scen_day_time}")
        print(f"{dataset.source}_{split}" +
              f"scenarios/s generation time = {scen_per_s}")
        # NOTE: [Debug-only] needed to check the scenarios/s speed
        scenarios_speed = {
            "100_day_total": total_gen,
            "100_day_mean": scen_day_time,
            "scenarios_s": scen_per_s}
        generated_scenarios = torch.cat(generated_scenarios, axis=0)

        # Save on disk
        if save:
            scenarios_name = (f"scenarios_{dataset.source}_" +
                f"{model.name}_{num_scenarios}_{split}.tar")
            scenarios_filepath = os.path.join(save_path, scenarios_name)
            torch.save(generated_scenarios, scenarios_filepath)

        return generated_scenarios, scenarios_speed

    def load_scenarios(self, model: pl.LightningModule, load_path: str,
                       num_scenarios: int=settings["data"]["num_scenarios"],
                       split: str="train") -> torch.Tensor:
        """ Load already generated power scenarios with the given model
        
        Parameters
        ----------
        - `model`: the (trained) generative model to sample scenarios
        - `load_path`: path where the scenarios are stored
        - `num_scenarios`: # scenarios to produce (per day)
        - `split`: the dataset split from which to make the generation

        Return
        ------
        `loaded_scenarios`: scenarios with power values for each day
        """    
        scenarios_name = (f"scenarios_{self.train_dataset.source}_" +
                          f"{model.name}_{num_scenarios}_{split}.tar")
        scenarios_filepath = os.path.join(load_path, scenarios_name)
        if len(os.listdir(load_path)) == 0:
            raise ValueError("There are no generated scenarios on disk")
        try:
            loaded_scenarios = torch.load(scenarios_filepath)
        except FileNotFoundError as fe:
            print("Exception =", fe)
            raise ValueError(
                "No scenarios available for the given parameters")
        return loaded_scenarios

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """ Dataloader for the training part """
        return torch.utils.data.DataLoader(self.gefcom_train, self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           drop_last=True, pin_memory=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """ Dataloader for the validation part """
        return torch.utils.data.DataLoader(self.gefcom_val, self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           drop_last=True, pin_memory=True)

    def test_dataloader(self) -> torch.utils.data.DataLoader: 
        """ Dataloader for the testing part """
        return torch.utils.data.DataLoader(self.gefcom_test, self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           drop_last=True, pin_memory=True)