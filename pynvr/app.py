""" parse command line, set up NVR and FastAPI application """
from asyncio import CancelledError
import json
import logging
import signal
import socket
from urllib.parse import urlparse, urlunparse

import click
import keyring
import uvicorn
from click import version_option
from passlib.context import CryptContext

from pynvr.api.endpoints import create_app
from pynvr.logger import setup_logging, KeywordFilter
from pynvr.nvr import NVR

NVR_OBJ = None

def shutdown(signum, _):
    """ called upon OS signal, set stop_event """
    NVR_OBJ.stop_event.set()
    logger.info(f"caught signal {signum}, stop event is set")
    NVR_OBJ.stop()

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

logger = logging.getLogger("pynvr")

def replace_url_credentials(url, new_username, new_password):
    """ parses RTSP url and replaces username password """
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

def socket_available(host, port):
    """ checks if socket is available """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            logger.error(f"socket {host}:{port} is already in use")
            return False
    return True

@click.command()
@click.option("-u", "--username", default="admin")
@click.option("-p", "--password", default="password://admin")
@click.option("--gui-username")
@click.option("--gui-password")
@click.option("-c", "--nvr-config", default="nvr.json")
@version_option()

# pylint: disable=too-many-branches, too-many-statements
def main(username,
         password,
         gui_username,
         gui_password,
         nvr_config):
    """ main entrypoint """

    #pylint: disable=global-statement
    global NVR_OBJ

    with open(nvr_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(config["logging_config"], encoding="utf-8") as f:
        logging_config_json = json.load(f)

    log_path = setup_logging(config["logging_config"])
    config["logs_directory"] = log_path

    logger.info("starting pynvr")

    if password.startswith("password://"):
        password = keyring.get_password(password, username)
        if password is not None:
            logger.debug(f"got password for {username} from keyring")

    if password is None:
        logger.error("password not set")
        return

    KeywordFilter.add_keyword(password)
    logger.info(f"password is set {password}")

    if gui_username is None:
        logger.info("using username as gui_username")
        gui_username = username

    if gui_password is None:
        logger.debug(f"using {username}'s password as gui_password")
        gui_password = password
    else:
        if gui_password.startswith("password://"):
            gui_password = keyring.get_password(gui_password, gui_username)
            logger.debug(f"got GUI password for {gui_username} from keyring")
        KeywordFilter.add_keyword(gui_password)
        logger.debug(f"GUI password is set {gui_password}")

    pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
    try:
        hashed_gui_password = pwd.hash(gui_password)
    except AttributeError:
        pass

    config["gui_username"] = gui_username
    config["gui_password"] = hashed_gui_password

    for camera in config["cameras"].values():
        camera["url"] = replace_url_credentials(camera["url"], username, password)

    if not socket_available(config["bind_address"], config["port"]):
        return

    NVR_OBJ = nvr = NVR(config)
    app = create_app(config, nvr)

    nvr.start()
    try:
        uvicorn.run(
            app,
            host=config["bind_address"],
            port=config["port"],
            log_config=logging_config_json,
            timeout_graceful_shutdown=0,
            access_log=False,
            workers=1,
            reload=False
            )
    except CancelledError:
        logger.debug("uvicorn server stopped")
    except Exception as e:
        logger.error(f"uvicorn server error {e}")

    logger.info("waiting on NVR threads to finish...")
    for thread in nvr.threads():
        logger.info(f"waiting on thread {thread.name}")
        thread.join()

    for handler in logger.handlers:
        handler.flush()
        handler.close()

    logger.info("done.")

if __name__ == "__main__":
    main()
