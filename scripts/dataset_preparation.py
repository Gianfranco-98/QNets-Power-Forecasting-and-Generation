# Fyle system
import os
import shutil
import zipfile

# Web resources
import urllib

# Personal files
from config import settings


# Global constants
ROOT = settings["path"]["root"]
DATAROOT = settings["path"]["dataroot"]


# Creating dataset dir
os.makedirs(DATAROOT, exist_ok=True)

# Downloading GEFCom_2014 dataset
if "Solar" not in os.listdir(DATAROOT) or "Wind" not in os.listdir(DATAROOT):
    print("Downloading and extracting GEFCom_2014 Dataset ...")
    a = input("")
    os.chdir(DATAROOT)
    # TODO: check if the following three lines are necessary in local
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(settings["data"]["url"], "GEFCom_2014.zip")
    with zipfile.ZipFile("GEFCom_2014.zip", "r") as zip_ref:
        zip_ref.extractall(os.getcwd())
    os.remove("GEFCom_2014.zip")
    for f in os.listdir("GEFCom2014 Data"):
        shutil.move(os.path.join(DATAROOT, "GEFCom2014 Data", f), DATAROOT)
    shutil.rmtree("GEFCom2014 Data")

    # Extracting GEFCom_2014 subfolders
    print("\tExtracting GEFCom_2014 subfolders ...")
    with zipfile.ZipFile("GEFCom2014-L_V2.zip", "r") as zip_ref:  
        zip_ref.extractall(os.getcwd())    # Load
    with zipfile.ZipFile("GEFCom2014-S_V2.zip", "r") as zip_ref:
        zip_ref.extractall(os.getcwd())    # Photovoltaic (Solar)
    with zipfile.ZipFile("GEFCom2014-W_V2.zip", "r") as zip_ref:
        zip_ref.extractall(os.getcwd())    # Eolic (Wind)

    # Extracting Wind data subfolders
    os.chdir(os.path.join(
        DATAROOT, "Wind", f"Task {settings['data']['task']}"))
    if (not os.path.isdir(settings["path"]["wind_power"]) or 
        not os.path.isdir(settings["path"]["wind_expvar"])):
        print("\t\tExtracting wind data subfolders ...")
        task_power_archive = \
            f"{os.path.basename(settings['path']['wind_power'])}.zip"
        task_expvar_archive = \
            f"{os.path.basename(settings['path']['wind_expvar'])}.zip"
        with zipfile.ZipFile(task_power_archive, "r") as zip_ref:
            zip_ref.extractall(os.getcwd())   # Wind main data
        with zipfile.ZipFile(task_expvar_archive, "r") as zip_ref:  
            zip_ref.extractall(os.getcwd())   # Wind remaining exp vars
        # Removing unnecessary archives
        os.remove(task_power_archive)
        os.remove(task_expvar_archive)

    # Removing unnecessary archives
    os.chdir(DATAROOT)
    os.remove("GEFCom2014-E_V2.zip")
    os.remove("GEFCom2014-L_V2.zip")
    os.remove("GEFCom2014-P_V2.zip")
    os.remove("GEFCom2014-S_V2.zip")
    os.remove("GEFCom2014-W_V2.zip")

os.chdir(ROOT)