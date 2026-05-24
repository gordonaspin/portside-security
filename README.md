# pynvr - a Python Network Video Recorder

`pynvr` is a capable NVR that records from IP camera streams over RTSP.

## Overview
`pynvr` uses ffmpeg to read RTSP streams. Each stream has its own ffmpeg subprocess that reads the stream and simultaneously writes to segment files and stdout. The segment files are not re-encoded, and the stdout output stream frames are converted to OpenCV2 format and resized to a frame size defined in the nvr.json config file. `pynvr` starts a thread per camera to read frames from the stdout stream and puts the latest frame to a per-camere queue. `pynvr` starts a second thread per camera to process the frame from the queue. Frame processing determines motion and object identificaation. When thresholds are met, recording is started. After a period of no motion, the recording is stopped and `pynvr` joins the segments together, re-encoding them to H.264.

`pynvr` is a server process that does not need a client to attach. The GUI implementation is svelte components and Javascript. `pynvr` GUI has controls to adjust motion-detection thresholds per camera. The GUI presents a mosaic of all enabled cameras, in the dimensions defined in nvr.json. If there are less cameras than rows * colums, black frames will be inserted. Clicking on a frame in the mosaic will zoom in to that camera, and clicking on a zoomed in camera will return to the mosaic.

The GUI presents a clickable timeline of events. Clicking the event will display the recorded video and associated metadata. The timeline can be panned back in history as far as the oldest event. The timeline can also be zoomed by holding the Shift-key and mouse-scroll. On mobile, the zoom is performed on two-finger scroll.
## Architecture / Design
`pynvr` implements an efficient, robust pipeline to stream per cameraa for motion and object detection and recording file creation. The pipeline is as follows:
```code
                         +---------------------------------------+
                         |     ffmpeg reader sub-process         +
                         | writes to mpeg segment files          +
                         | writes raw downsized frames to stdout +
                         +---------------------------------------+
                                            |
                                            v
                         +---------------------------------------+
                         |         frame reader thread           +
                         | reads frames from stdout              +
                         | enqueues frame, lastest frame wins    +
                         +---------------------------------------+
                                            |
                                            v
                         +---------------------------------------+      +--------------------------------------+
                         |       frame processor thread          +      +            GUI threads               +
                         | gets frame from queue                 + <--- + uses frame from memory               +
                         | detects motion and objects.           +      + renders frame to GUI                 +
                         +---------------------------------------+      +--------------------------------------+
                                            |
                                            v
                         +---------------------------------------+
                         |     ffmpeg merge sub-process          +
                         | asynchronously merges mpeg segment    +
                         | files and reencodes to H264 MP4 file  +
                         +---------------------------------------+

```
## Installation
Clone the repo
```bash
git clone http://github.com/gordonaspin/pynyr.git
```
Install the required python libraries
```bash
pip install -r requirements.txt
```
Install nodejs and npm, if needed
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```
Install project dependencies
```bash
cd frontend
npm install
```
Build the frontend GUI application
```bash
npm run build
```
## Usage - Command line options

#### -c | --nvr-config filename.json
Optional argument to specify the NVR configuration file. Default is nvr.json. See config section for JSON format spec.
#### -u username -p password
If supplied `pynvr` will apply those credentials to the RTSP urls specified in your NVR config file. If not supplied, `pyvr` will use the credentials from the RTSP URLs in the NVR config file. If the password begins with "password://" `pynvr` will retrieve the password from the python keyring using password://<username>.
#### --gui-username username --gui_password password
If supplied `pynvr` will apply these credentials to the GUI and will present a login challenge that accepts these credentials only. If not supplied `pynvr` uses the same credentials as username/password.
```bash
python backend/app.py -u rtsp-username -p rtsp-password
```
## Config
Configuration is provided in a nvr.json file. "resolution" specifies the [x, y] dimensions in pixels to resize frames to for YOLO processing and rendering on the GUI. "yolo.model" specifies the name of the YOLO model to use. "yolo.classes" is an array of coco classes of objects to detect in the image processing. Each camera is named and specifies the RTSP URL and per-camera motion detection parameters, and enabled and debug flags.
```json
{
    "system_name": "My Cameras",                          # displayed on the GUI
    "recordings_directory": "recordings",                 # folder to store recordings, metadata etc.
    "resolution": {                                       # resolution to re-scale the video from the camera for motion detection
        "width": 704,
        "height": 480
    },
    "bind_address": "0.0.0.0",                            # IP address to bind the API/GUI server
    "port": 7860,                                         # port number to serve the GUI
    "logging_config": "logging-config.json",              # python logging configuration
    "debug": false,
    "mosaic": {                                           # dimensions of the mosaic frames
        "rows": 2,
        "columns": 5
    },
    "yolo": {
        "classes": ["person", "car", "truck", "bus", "cat", "dog", "bicycle", "motorcycle"],
        "model": "backend/model/yolov8n.pt",
        "lpr_model": "backend/model/license-plate-finetune-v1l.pt"
    },
    "cameras": {
        "Cam1": {
            "url": "rtsp://username:password@hostname.com:554/cam/realmonitor?channel=3&subtype=1",
            "enabled": true,
            "yolo_confidence": 0.5,                                      # 0.1-0.9 confidence
            "motion_threshold": 0.4,                                     # 0.1%-0.9% of total pixels 
            "minimum_motion_confidence": 0.3,                            # 0.1-0.9 confidence
            "minimum_motion_frames": 8.0,                                # number of frames with motion required to begin recording
            "minimum_sum_box_area": 0.7,                                 # 0.1%-1.2% of total pixels to consider motion
            "debug": false
        },
        "Cam2": {
            "url": "rtsp://username:password@hostname.com:554/cam/realmonitor?channel=4&subtype=1",
            "enabled": true,
            "yolo_confidence": 0.5,
            "motion_threshold": 0.4,
            "minimum_motion_confidence": 0.3,
            "minimum_motion_frames": 8.0,
            "minimum_sum_box_area": 0.7,
            "debug": false
        },
        "Cam3": {
            "url": "rtsp://username:password@hostname.com:554/cam/realmonitor?channel=5&subtype=1",
            "enabled": true,
            "yolo_confidence": 0.5,
            "motion_threshold": 0.4,
            "minimum_motion_confidence": 0.3,
            "minimum_motion_frames": 8.0,
            "minimum_sum_box_area": 0.7,
            "debug": false
        },
        "Entry": {
            "url": "rtsp://username:password@hostname.com:554/cam/realmonitor?channel=2&subtype=1",
            "enabled": true,
            "yolo_confidence": 0.5,
            "motion_threshold": 0.4,
            "minimum_motion_confidence": 0.3,
            "minimum_motion_frames": 8.0,
            "minimum_sum_box_area": 0.7,
            "debug": false
        },
        "Exit": {
            "url": "rtsp://username:password@hostname.com:554/cam/realmonitor?channel=1&subtype=1",
            "enabled": true,
            "yolo_confidence": 0.5,
            "motion_threshold": 0.4,
            "minimum_motion_confidence": 0.3,
            "minimum_motion_frames": 8.0,
            "minimum_sum_box_area": 0.7,
            "lpr": {
                "enabled": false,
                "url": "rtsp://username:password@hostname.com:554/cam/realmonitor?channel=1&subtype=0",
                "top": 720,
                "left": 384,
                "width": 2304,
                "height": 760
            },
            "debug": false
        }
    }
}
```
### Detailed usage
```bash
Usage: app.py -d <directory> -u -p [options]

Options:
  -c, --nvr-config <file>         NVR config file  [defaults to nvr.json]
  -u, --username <username>       NVR/Camera username, will override what is specified in the --nvr-config file
  -p, --password <password>       NVR/Camera password, will override what is specified in the --nvr-config file
  --gui-username <GUI username>   GUI authorization username
  --gui-password <GUI password>   GUI authorization password
  --version                       Show the version and exit.
  -h, --help                      Show this message and exit.
```
### Video Frames
`pynvr` displays yolo boxes when there is motion that overlaps or includes a recognized object. `pynvr` draws the current state of recording and fps information on the top left of the frame.

### Recordings and Metadata
`pynvr` creates recordings using ffmpeg .ts segment files. `pynvr` creates .json metadata for each recording which details the start/stop time, the objects captured, etc.

### Logging
`pynvr` uses python logging to write to log files, configured by the logging-config.json file. Passwords passed in on the command line or fetch from the keyring are filtered from logs. `pynvr` also produces log files per camera attached to each of the ffmpeg sub-processes and a log file per recording merge.
