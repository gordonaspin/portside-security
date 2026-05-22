import atexit
import signal
import sys
import json
import logging
from urllib.parse import urlparse, urlunparse

import click
import keyring
from click import version_option
from passlib.context import CryptContext
import uvicorn

from api.endpoints import create_app

import constants as constants
from context import Context
from logger.logger import setup_logging, KeywordFilter
from nvr.nvr import NVR

_NVR = None

def shutdown(signum, frame):
    _NVR.stop_event.set()
    _NVR.stop()

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

logger = logging.getLogger("nvr")

def replace_url_credentials(url, new_username, new_password):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""

    userinfo = ""
    if new_username is not None:
        userinfo = new_username
        if new_password is not None:
            userinfo += f":{new_password}"
        userinfo += "@"

    new_netloc = f"{userinfo}{hostname}{port}"
    new_parsed = parsed._replace(netloc=new_netloc)
    return urlunparse(new_parsed)


@click.command()
@click.option("-d", "--directory", required=True)
@click.option("-c", "--nvr-config", default="nvr.json")
@click.option("-u", "--username", default="admin")
@click.option("-p", "--password", default="password://admin")
@click.option("--gui-username")
@click.option("--gui-password")
@click.option("--bind-address", default="0.0.0.0")
@click.option("--logging-config", default="logging-config.json")
@click.option("--debug", is_flag=True)
@version_option()
def main(directory, username, password, gui_username, gui_password,
         nvr_config, bind_address, logging_config,
         debug):

    global _NVR
    with open(logging_config, encoding="utf-8") as f_in:
        logging_config_json = json.load(f_in)

    log_path = setup_logging(logging_config)

    with open(nvr_config, "r") as f:
        config = json.load(f)

    resolution = config["resolution"]
    camera_config = config["cameras"]
    
    if password.startswith("password://"):
        password = keyring.get_password(password, username)
        logger.debug(f"got password for {username} from keyring")
    if password is None:
            return

    if password:
        KeywordFilter.add_keyword(password)
        logger.debug(f"got password {password}")

    if gui_username is None:
        logger.debug(f"using username as gui_username")
        gui_username = username

    if gui_password is None:
        logger.debug(f"using {username}'s password as gui_password")
        gui_password = password
    else:
        if gui_password.startswith("password://"):
            gui_password = keyring.get_password(gui_password, gui_username)
            logger.debug(f"got GUI password for {gui_username} from keyring")
        KeywordFilter.add_keyword(gui_password)
        logger.debug(f"got GUI password {gui_password}")

    pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
    try:
        hashed_gui_password = pwd.hash(gui_password)
    except AttributeError:
        pass

    for cam in camera_config.values():
        cam["url"] = replace_url_credentials(cam["url"], username, password)
        try:
            cam["lpr"]["url"] = replace_url_credentials(cam["lpr"]["url"], username, password)
        except KeyError:
            pass

    ctx = Context(
        directory=directory,
        log_directory=log_path,
        username=username,
        password=password,
        gui_username=gui_username,
        gui_password=hashed_gui_password,
        camera_config=camera_config,
        bind_address=bind_address,
        resolution=resolution,
        yolo_config=config["yolo"],
        debug=debug,
    )

    _NVR = nvr = NVR(ctx)
    nvr.start()
    atexit.register(nvr.stop)

    app = create_app(ctx, nvr)
    uvicorn.run(app, host=ctx.bind_address, port=7860, log_config=logging_config_json, access_log=False)
    logger.info("Exiting")

if __name__ == "__main__":
    main()
