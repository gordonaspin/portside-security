# pynvr - a Python Network Video Recorder

`pynvr` is a capable NVR that records from IP camera streams over RTSP.

## Overview
`pynvr` uses ffmpeg to read RTSP streams. Each stream has its own ffmpeg subprocess that reads the stream and simultaneously writes frames to stdout and optionally FFmpeg segment files. The segment files are not re-encoded. The stdout output stream frames are converted to OpenCV2/bgr24 format and resized to a frame size defined in the nvr.json config file. `pynvr` starts a thread per camera to read frames from the stdout stream and puts the latest frame to a per-camera queue. `pynvr` starts a second thread per camera to process the frame from the queue. Frame processing determines motion and object identificaation. When thresholds are met, recording is started. After a period of no motion, the recording is stopped and `pynvr` writes frames to a video file or joins the segments together, re-encoding them to H.264.

`pynvr` is a server process that does not need a client to attach. The GUI implementation is svelte components and Javascript. The `pynvr` GUI has controls to adjust motion-detection thresholds per camera. The GUI presents a mosaic of all enabled cameras, in the dimensions defined in nvr.json. If there are less cameras than rows * colums, black frames will be inserted. Clicking on a frame in the mosaic will zoom in to that camera, and clicking on a zoomed in camera will return to the mosaic.

The GUI presents a clickable timeline of events. Clicking the event will display the recorded video and associated metadata. The timeline can be panned back in history as far as the oldest event. The timeline can also be zoomed by holding the Shift-key and mouse-scroll. On mobile, the pan and zoom is performed by one finger and two-finger scroll, respectively.

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
Install external dependencies
```
ffmpeg
```
## Usage - Command line options

#### -c | --nvr-config filename.json
Optional argument to specify the NVR configuration file. Default is nvr.json. See config section for JSON format spec.
#### -u username -p password
If supplied `pynvr` will apply those credentials to the RTSP urls specified in your NVR config file. If not supplied, `pyvr` will use the credentials from the RTSP URLs in the NVR config file. If the password begins with "password://" `pynvr` will retrieve the password from the python keyring using password://<username>.
#### --gui-username username --gui_password password
If supplied `pynvr` will apply these credentials to the GUI and will present a login challenge that accepts these credentials only. If not supplied `pynvr` uses the same credentials as username/password.
```bash
python pynvr/app.py -u rtsp-username -p rtsp-password
```
## Docker container usage
`pynvr` can be built as a docker image, using the provided script which you customize to your needs.
```bash
build.docker.sh
```
The configuration files, nvr.json and logging-config.json are not built into the container. The `pynvr` application and gui frontend are in the container. It is necessary to expose port 7860 to the host and map a host folder for the configuration files and recordings folder., e.g.:
```bash
docker run -it -d --restart=always --network host --ipc=host --gpus all --name pynvr -v path/to/a/folder:/home/docker dockerimage <pynvr-options> 
```

## Config
Configuration is provided in a nvr.json file. "model.resolution" specifies the [x, y] dimensions in pixels to resize frames to for YOLO processing and rendering on the GUI. "model.name" specifies the name of the YOLO model to use. "model.classes" is an array of coco names of object classes to detect in the image processing. Each camera is named and specifies the RTSP URL and per-camera resolution, motion detection parameters, enabled and debug flags and which recorder type to use.
```json
{
    "system_name": "My Cameras",
    "recordings_directory": "recordings",
    "keep_recordings_timedelta": {
        "days": 3
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
            "width": 640,
            "height": 640
        },
        "classes": {"person": true, "car": true, "truck": true, "bus": true, "cat": true, "dog": true, "bicycle": true, "motorcycle": true},
        "_comment": "yolo11n.pt  yolo11s.pt  yolov5su.pt yolov8n.pt",
        "name": "model/yolo11n.pt"
    },
    "processor": {
        "detect_every_nth_frame": 1,
        "device": "cuda",
        "night_check_period": 5,
        "recorder": {
            "startup_delay": 15,
            "pre_duration": 3,
            "post_duration": 3
        }
    },
    "cameras": {
        "B1": {
            "enabled": true,
            "url": "rtsp://username:password@hostname.com:554/cam/realmonitor?channel=3&subtype=1",
            "resolution": {
                "width": 704,
                "height": 480
            },
            "_recorder_comment": "FFmpegSegment or FFmpegFrame or OpenCVFrame or AVFFmpegFrame",
            "recorder": "FFmpegFrame",
            "_yolo_confidence_comment": "0.1-0.9 confidence",
            "yolo_confidence": 0.4,
            "track_threshold": 0.35,
            "match_threshold": 0.4,
            "track_buffer": 120,
            "minimum_relative_motion": 0.08, 
            "_render_annotations_comment": "always or never or motion",
            "render_annotations": "always",
            "debug": false
        },
        ...
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
`pynvr` displays yolo boxes when there is motion that includes a recognized object. `pynvr` displays the current state of recording and fps information on the top left of the frame in the UI.

### Recordings and Metadata
`pynvr` creates recordings using ffmpeg .ts segment files or frames. `pynvr` creates .json metadata for each recording which details the start/stop time, the objects captured, etc.

### Recorder types
`pynvr` has 4 recorder types; FFmpegSegment, FFmpegFrame, OpenCVFrame and AVFFmpegFrame.
#### FFMpegSegment
The FFMpegSegment recorder makes recordings from the segment files produced by FFMpeg. It is a recording of what the camera saw without any overlays indicating objects seen moving.
#### FFmpegFrame or OpenCVFrame or AVFFmpegFrame
FFmpegFrame or OpenCVFrame or AVFFmpegFrame are functionally equivalent, using different technologies. Each records frames that include overlays of object classes detected and yolo markup etc. The FFMpegFrame recorder passes frames at their recorded pace to FFmpeg over stdin. The OpenCVFrame recorder uses OpenCV to create the mp4 video file followed by an FFMpeg transformation to make the mp4 streamable. The AVFFmpegFrame recorder uses PyAV to create the recording from the frames, which uses FFMpeg under the covers.

### Logging
`pynvr` uses python logging to write to log files, configured by the logging-config.json file. Passwords passed in on the command line or fetch from the keyring are filtered from logs. `pynvr` also produces log files per camera attached to each of the ffmpeg sub-processes and a log file per recording merge.

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
                         |       frame processor thread          +      + Svelte UI and WebRTC GUI uses frame  +
                         | gets frame from queue                 + <--- + from memory and renders frame to GUI +
                         | detects motion and objects.           +      + receives realtime events from server +
                         +---------------------------------------+      +--------------------------------------+
                                            |
                                            v
                         +---------------------------------------+
                         |              recorders.               +
                         | asynchronously merge mpeg segment     +
                         | files or frames and create MP4        +
                         | recording, create metadata            +
                         +---------------------------------------+

```
### Technologies used
`python` is used to create the pynvr server and all the video processing with the exception of ffmpeg.
`FastAPI and uvicorn` is used as the API to the server, by default it listens on port 7860.
`ffmpeg` is used to connect to IP cameras using the url configured for the camera in nvr.json. `ffprobe` is used to detect the native resolution of the stream if possible, and that resolution will be used to stream video to the user-interface and make recordings. If the native resolution of the camera cannot be determined by ffprobe, the fallback is the configured resolution. If the resolution is different from the YOLO resolution, then a second rawvideo is output from ffmpeg at the configured resolution will be used for object detection.
`YOLO` is used for object detection and classification. The expected model is yolo11n, and can be changed in the configuration file. YOLO processing will use the GPU for processing, if available and the configured processor.device is not "cpu". If "cpu" is specified, the processor.detect_every_nth_frame should be adjusted upwards to every 3rd or 4th frame, depending on your CPU.
`OpenCV` is used for frame manipulation, blurring, grayscaling etc. as needed to detect and measure motion
`PyAV` is used (with ffmpeg under the covers) to make recordings from frames.
`Svelte` (typescript/javascript) is used to create the user interface with Javascript, HTML markup and CSS.
`WebRTC` is used to stream video content to the Mosaic.svelte component in the browser.
`SSE or Server Side Events` is used to push updates to the browser. It is used for camera status displayed as a <div> overlay on the camera image in the Mosaic.svelte component. It is also used to push server logs to the browser EventLog component and recording event metadata to the browser that are subsequently fed to the EventInfo.svelte component and the MediaPlayer component.