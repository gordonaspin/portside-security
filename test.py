import subprocess
import numpy as np
import cv2

rtsp_url = "rtsp://admin:Portside123!!!@shed.portsidecondominium.com:554/cam/realmonitor?channel=1&subtype=1"

width = 704
height = 480
ffmpeg_cmd4 = [
    "ffmpeg",

    "-rtsp_transport", "tcp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-use_wallclock_as_timestamps", "1",
    "-i", rtsp_url,

    # Split + per-branch processing
    "-filter_complex",
    f"[0:v]split=2[v1][v2];"
    f"[v1]scale={width}:{height}[enc];"
    f"[v2]scale={width}:{height},format=bgr24[raw]",

    # ---- TS segments (encoded) ----
    "-map", "[enc]",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-tune", "zerolatency",
    "-g", "30",
    "-keyint_min", "30",
    "-sc_threshold", "0",
    "-f", "segment",
    #"-segment_time", "1",
    "-reset_timestamps", "1",
    "-strftime", "1",
    "-segment_format", "mpegts",
    "test%Y%m%d_%H%M%S.ts",

    # ---- Raw frames (OpenCV) ----
    "-map", "[raw]",
    "-f", "rawvideo",
    "pipe:1"
]
ffmpeg_cmd3 = [
    "ffmpeg",

    "-rtsp_transport", "tcp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-i", rtsp_url,

    # Split + per-branch processing
    "-filter_complex",
    f"[0:v]split=2[v1][v2];"
    f"[v1]scale={width}:{height}[enc];"
    f"[v2]scale={width}:{height},format=bgr24[raw]",

    # ---- TS segments (encoded) ----
    "-map", "[enc]",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-tune", "zerolatency",
    "-f", "segment",
    #"-segment_time", "1",
    "-reset_timestamps", "1",
    "-strftime", "1",
    "-segment_format", "mpegts",
    "test%Y%m%d_%H%M%S.ts",

    # ---- Raw frames (OpenCV) ----
    "-map", "[raw]",
    "-f", "rawvideo",
    "pipe:1"
]
ffmpeg_cmd2 = [
    "ffmpeg",

    # RTSP input options
    "-rtsp_transport", "tcp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-i", rtsp_url,

    # ✅ CRITICAL: map the video stream
    "-map", "0:v:0",

    # Video processing for OpenCV
    "-vf", f"scale={width}:{height}",
    "-pix_fmt", "bgr24",

    # Encode for segment output (TS-friendly)
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-tune", "zerolatency",

    "-f", "tee",

    # Tee outputs:
    # 1. TS segments
    # 2. Raw frames to stdout
    "[f=segment:segment_time=5:reset_timestamps=1:"
    "segment_format=mpegts]output_%03d.ts|"
    "[f=rawvideo]pipe:1"
]
ffmpeg_cmd = [
    "ffmpeg",

    # RTSP input options (important!)
    "-rtsp_transport", "tcp",        # more reliable than UDP
    "-fflags", "nobuffer",           # reduce latency
    "-flags", "low_delay",
    "-i", rtsp_url,

    # Normalize video for OpenCV
    "-vf", f"scale={width}:{height}",
    "-pix_fmt", "bgr24",

    "-f", "tee",

    # Tee outputs:
    # 1. Segment files (H.264 encoded)
    # 2. Raw frames to stdout
    "[f=segment:segment_time=5:reset_timestamps=1:strftime=1]"
    "test%Y%m%d_%H%M%S.ts|"
    "[f=rawvideo]pipe:1"
]

process = subprocess.Popen(
    ffmpeg_cmd4,
    stdout=subprocess.PIPE,
#    stderr=subprocess.PIPE,
    bufsize=10**8
)

frame_size = width * height * 3

while True:
    raw_frame = process.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        break

    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

    cv2.imshow("RTSP Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

process.stdout.close()
process.wait()
cv2.destroyAllWindows()
