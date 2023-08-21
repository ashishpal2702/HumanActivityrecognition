import os
import toml
import json
import logging
import pandas as pd
import joblib
from datetime import datetime

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


def load_config():
    """

    @return:  toml file configurations
    @rtype: object
    """
    config_file = "config.toml"
    filepath = os.path.join(PROJECT_DIR, "config", config_file)
    with open(filepath, "r") as f:
        return toml.load(f)


def update_log(message):
    """
    Function to update info log
    Args:
        secret: message want to save
    Returns: NA
    """
    log_folder = "log"
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.basicConfig(
        filename="log/log_info.log", format="%(asctime)s %(message)s", filemode="a+"
    )
    # Creating an object
    logger = logging.getLogger()
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.INFO)

    logger.info(message)

def save_weights(obj, file):
    filepath = os.path.join(PROJECT_DIR, file)
    joblib.dump(obj, filepath)

def load_weights(file):
    filepath = os.path.join(PROJECT_DIR, file)
    object = joblib.load(filepath)
    return object


def save_model(model):

    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d %H:%M:%S")
    dir_path = os.path.join(PROJECT_DIR, "model", type(model).__name__, str(date_time))
    if not os.path.exists(dir_path):
        print("Creating model Dir at, ", dir_path)
        os.makedirs(dir_path)
    filepath = os.path.join(dir_path, "model.pkl")
    joblib.dump(model, filepath)
