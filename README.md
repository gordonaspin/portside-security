# pynvr - a Python Network Video Recorder

`pynvr` is a capable NVR that records from IP camera streams over RTSP.

`pynvr` uses ffmpeg to read RTSP streams. Each stream has its own ffmpeg subprocess that reads the stream and simultaneously writes to segment files and stdout. The segment files are not re-encoded, and the stdout output stream frames are converted to OpenCV2 format and resized to a frame size defined in the nvr.json config file. pynvr starts a thread per camera to read frames from the stdout stream and puts the latest frame to a per-camere queue. pynvr starts a second thread per camera to process the frame from the queue. Frame processing determines motion and object identificaation. When thresholds are met, recording is started. After a period of no motion, the recording is stopped and pynvr joins the segments together, re-encoding them to H.264.

`pynvr` has a user interface with controls to adjust thresholds and objects to be detected. pynvr is a server process that does not need a client to attach. The GUI implementation uses Gradio to render frames from the cameras. The GUI presents rows of up to 5 cameras per row, a log window and and list of hyperlinks to recordings.
## Installation
Clone the repo
```bash
git clone http://github.com/gordonaspin/pynyr.git
```
Install the required python libraries
```bash
pip install -r requirements.txt
```
## Usage
The minimum required arguments are a pre-existing folder to write recordings to, and the credentials required for the camera RTSP streams. `pynvr` assumes all cameras use the same credentials. For other options, see the section on detailed usage.
```bash
python app.py -d <recordings folder> -u rtsp-username -p rtsp-password
```
## Config
Configuration is provided in a nvr.json file, specifying the YOLO model to use, the resolutiom of render images to resize to, and the RTSP urls of each camera. The username and password tokens are replaced by the values provided on the command line by pynvr at startup.
```json
{
    "downsize_resolution": [704, 480],
    "yolo": {
        "classes": ["person", "car", "truck", "bus", "bicycle", "motorcycle", "cat", "dog"],
        "model": "yolov8n.pt"
    },
    "cameras": {
        "Cam1": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=3&subtype=1",
            "enabled": true
        },
        "Cam2": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=4&subtype=1",
            "enabled": true
        },
        "Cam3": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=5&subtype=1",
            "enabled": true
        },
        "Cam4": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=2&subtype=1",
            "enabled": true
        },
        "Cam5": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=1&subtype=1",
            "enabled": true
        }
    }
}
```
### Detailed usage
```bash
Usage: app.py -d <directory> -u -p [options]

Options:
  -d, --directory <directory>     Local directory that should be used for storing recordings and logs  [required]
  -u, --username <username>       NVR username  [required]
  -p, --password <password>       NVR password  [required]
  -c, --nvr-config <file>         NVR config file  [required]
  --bind-address <ip address>     bind address for gradio GUI  [default: 0.0.0.0]
  --logging-config <filename>     JSON logging config filename (default: logging-config.json)  [default: logging-
                                  config.json]
  --motion-threshold <threshold>  threshold for motion detection (day, night)  [default: 500, 4000]
  --confidence-threshold <threshold>
                                  Confidence threshold for object detection  [default: 0.3; 0.1<=x<=0.9]
  --motion-detect-frame-count <seconds>
                                  number of frames with motion required to start recording  [default: 40; 2<=x<=100]
  --debug                         debug mode, produces .jpg files of motion contours
  --no-auth                       don't present the login screen
  --subtype INTEGER RANGE         rtsp subtype override  [default: 1; 0<=x<=2]
  --version                       Show the version and exit.
  -h, --help                      Show this message and exit.
```
### Logging
`pynvr` uses python logging to write to log files, configured by the logging-config.json file. The password passed in on the command line is filtered from logs. `pynvr` also produces log files per camera attached to each of the ffmpeg sub-processes and a log file per recording merge.