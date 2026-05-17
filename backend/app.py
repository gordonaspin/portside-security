import atexit
import signal
import sys
import json
import logging
from urllib.parse import urlparse, urlunparse

import click
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
@click.option("-u", "--username")
@click.option("-p", "--password")
@click.option("--gui-username")
@click.option("--gui-password")
@click.option("--bind-address", default="0.0.0.0")
@click.option("--logging-config", default="logging-config.json")
@click.option("--motion-threshold", default=constants.MOTION_THRESHOLD)
@click.option("--confidence-threshold", default=constants.CONFIDENCE_THRESHOLD)
@click.option("--debug", is_flag=True)
@version_option()
def main(directory, username, password, gui_username, gui_password,
         nvr_config, bind_address, logging_config,
         motion_threshold, confidence_threshold,
         debug):

    global _NVR

    log_path = setup_logging(logging_config)
    if password:
        KeywordFilter.add_keyword(password)

    with open(nvr_config, "r") as f:
        config = json.load(f)

    yolo_config = config["yolo"]
    resolution = config["resolution"]
    camera_config = config["cameras"]

    if username and password:
        for cam in camera_config.values():
            cam["url"] = replace_url_credentials(cam["url"], username, password)
            try:
                cam["lpr"]["url"] = replace_url_credentials(cam["lpr"]["url"], username, password)
            except KeyError:
                pass

    pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
    try:
        hashed_gui_password = pwd.hash(gui_password)
    except AttributeError:
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
        motion_threshold=motion_threshold,
        confidence_threshold=confidence_threshold,
        resolution=resolution,
        model=yolo_config["model"],
        lpr_model=yolo_config["lpr_model"],
        classes=yolo_config["classes"],
        debug=debug,
    )

    _NVR = nvr = NVR(ctx)
    nvr.start()
    atexit.register(nvr.stop)

    app = create_app(ctx, nvr)
    uvicorn.run(app, host=ctx.bind_address, port=7860, log_level="info")
    logger.info("Exiting")

if __name__ == "__main__":
    main()
