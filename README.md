# pynvr - a Python Network Video Recorder

`pynvr` is a capable NVR that records from IP camera streams over RTSP.

## Overview
`pynvr` uses ffmpeg to read RTSP streams. Each stream has its own ffmpeg subprocess that reads the stream and simultaneously writes frames to stdout and optionally FFmpeg segment files. The segment files are not re-encoded, and the stdout output stream frames are converted to OpenCV2/bgr24 format and resized to a frame size defined in the nvr.json config file. `pynvr` starts a thread per camera to read frames from the stdout stream and puts the latest frame to a per-camere queue. `pynvr` starts a second thread per camera to process the frame from the queue. Frame processing determines motion and object identificaation. When thresholds are met, recording is started. After a period of no motion, the recording is stopped and `pynvr` writes frames to a video file or joins the segments together, re-encoding them to H.264.

`pynvr` is a server process that does not need a client to attach. The GUI implementation is svelte components and Javascript. The `pynvr` GUI has controls to adjust motion-detection thresholds per camera. The GUI presents a mosaic of all enabled cameras, in the dimensions defined in nvr.json. If there are less cameras than rows * colums, black frames will be inserted. Clicking on a frame in the mosaic will zoom in to that camera, and clicking on a zoomed in camera will return to the mosaic.

The GUI presents a clickable timeline of events. Clicking the event will display the recorded video and associated metadata. The timeline can be panned back in history as far as the oldest event. The timeline can also be zoomed by holding the Shift-key and mouse-scroll. On mobile, the pand and zoom is performed by one finger and two-finger scroll, respectively.
## Architecture / Design
`pynvr` implements an efficient, robust pipeline per camera for motion and object detection and recording file creation. The pipeline is as follows:
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
                         |       frame processor thread          +      +        Svelte UI and WebRTC GUI      +
                         | gets frame from queue                 + <--- +        uses frame from memory        +
                         | detects motion and objects.           +      +        renders frame to GUI          +
                         +---------------------------------------+      +--------------------------------------+
                                            |
                                            v
                         +---------------------------------------+
                         |              recorders.               +
                         | asynchronously merges mpeg segment    +
                         | files or frames and creates MP4       +
                         | recording, creates metadata           +
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
Configuration is provided in a nvr.json file. "model.resolution" specifies the [x, y] dimensions in pixels to resize frames to for YOLO processing and rendering on the GUI. "model.name" specifies the name of the YOLO model to use. "model.classes" is an array of coco names of object classes to detect in the image processing. Each camera is named and specifies the RTSP URL and per-camera resolution, motion detection parameters, enabled and debug flags and which recorder type to use.
```json
{
    "system_name": "Portside Cameras",
    "recordings_directory": "recordings",
    "keep_recordings_timedelta": {
        "days": 7
    },
    "keep_logs_timedelta": {
        "days": 1
    },
    "bind_address": "0.0.0.0",
    "port": 7860,
    "logging_config": "logging-config.json",
    "debug": false,
    "mosaic": {
        "rows": 2,
        "columns": 5
    },
    "model": {
        "resolution": {
            "width": 704,
            "height": 480
        },
        "classes": ["person", "car", "truck", "bus", "cat", "dog", "bicycle", "motorcycle"],
        "name": "backend/model/yolov8n.pt"
    },
    "cameras": {
        "B1": {
            "enabled": true,
            "url": "rtsp://username:password@portsideb3.noip.me:554/cam/realmonitor?channel=3&subtype=1",
            "resolution": {
                "width": 704,
                "height": 480
            },
            "_recorder_comment": "FFmpegSegment or FFmpegFrame or OpenCVFrame or AVFFmpegFrame",
            "recorder": "AVFFmpegFrame",
            "_yolo_confidence_comment": "0.1-0.9 confidence",
            "yolo_confidence": 0.5,
            "_motion_threshold_comment": "0.1%-0.9% pixels",
            "motion_threshold": 0.4,
            "_minimum_motion_confidence_comment": "0.1-0.9 confidence",
            "minimum_motion_confidence": 0.3,
            "_day_motion_frames_comment": "5-20 frames",
            "minimum_motion_frames": 4.0,
            "_minimum_sum_box_area_comment": "0.1%-1.2% pixels",
            "minimum_sum_box_area": 0.7,
            "debug": false
        },
        "B2": {
            "enabled": true,
            "url": "rtsp://username:password@portsideb3.noip.me:554/cam/realmonitor?channel=4&subtype=1",
            "resolution": {
                "width": 704,
                "height": 480
            },
            "recorder": "AVFFmpegFrame",
            "yolo_confidence": 0.5,
            "motion_threshold": 0.4,
            "minimum_motion_confidence": 0.3,
            "minimum_motion_frames": 4.0,
            "minimum_sum_box_area": 0.7,
            "debug": false
        },
        "B3": {
            "enabled": true,
            "url": "rtsp://username:password@portsideb3.noip.me:554/cam/realmonitor?channel=5&subtype=1",
            "resolution": {
                "width": 704,
                "height": 480
            },
            "recorder": "AVFFmpegFrame",
            "yolo_confidence": 0.5,
            "motion_threshold": 0.4,
            "minimum_motion_confidence": 0.3,
            "minimum_motion_frames": 4.0,
            "minimum_sum_box_area": 0.7,
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
`pynvr` creates recordings using ffmpeg .ts segment files or frames. `pynvr` creates .json metadata for each recording which details the start/stop time, the objects captured, etc.

### Recorder types
`pynvr` has 4 recorder types; FFmpegSegment, FFmpegFrame, OpenCVFrame and AVFFmpegFrame.
#### FFMpegSegment
The FFMpegSegment recorder makes recordings from the segment files produced by FFMpeg. It is a recording of what the camera saw without any overlays indicating objects seen moving.
#### FFmpegFrame or OpenCVFrame or AVFFmpegFrame
FFmpegFrame or OpenCVFrame or AVFFmpegFrame are functionally equivalent, just different technologies used. Each records frames that include overlays of processing status such as object classes detected and yolo markup etc. The FFMpegFrame recorder passes frames at their recorded pace to FFmpeg over stdin. The OpenCVFrame recorder uses OpenCV to create the mp4 video file followed by an FFMpeg transformation to make the mp4 streamable. The AVFFmpegFrame recorder uses PyAV to create the recording from the frames, which uses FFMpeg under the covers.

### Logging
`pynvr` uses python logging to write to log files, configured by the logging-config.json file. Passwords passed in on the command line or fetch from the keyring are filtered from logs. `pynvr` also produces log files per camera attached to each of the ffmpeg sub-processes and a log file per recording merge.
