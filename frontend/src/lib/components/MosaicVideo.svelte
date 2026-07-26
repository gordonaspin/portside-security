<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { log, error } from "$lib/stores/logging";
  import { safeFetch } from '$lib/network/safeFetch';
  import { cameraStatusStore } from "$lib/stores/cameraStatus";
  import { serverOffline } from '$lib/stores/connection';

  let cameras = [];
  let isFocusMode = false;
  let currentCamera = null;

  let mosaicPC = null;
  let focusPC = null;

  let mosaicVideo; // bind:this
  let mosaicTitle = "";
  let mosaicRows = 0;
  let mosaicCols = 0;
  let videoWidth = 3840;
  let videoHeight = 1046;


  async function loadCameras() {
    try {
      log("Loading cameras...");
      const res = await safeFetch("/api/cameras", { credentials: "include" });
      cameras = await res.json();
    } catch (err) {
      error("Failed to load cameras:", err);
    }
  }

  async function loadMosaicDimensions() {
    try {
      const res = await safeFetch("/api/mosaic_dimensions", { credentials: "include" });
      const dimensions = await res.json();
      mosaicRows = dimensions.rows;
      mosaicCols = dimensions.columns;
      log("Loading mosaic dimensions:", dimensions);

    } catch (err) {
      error("Failed to load mosaic dimensions:", err);
    }
  }

  function attachVideoTrack(videoEl, stream) {
    if (!videoEl) return;

    log("Attaching video stream:", stream);
    videoEl.srcObject = stream;
    videoEl.muted = true;
    videoEl.play().catch(() => {});
  }

  function setVideoDimensions(settings) {
    videoWidth = settings.width;
    videoHeight = settings.height;
  }

  async function startMosaic() {
    mosaicTitle = "All Cameras";

    log("Starting mosaic view");
    if (focusPC) {
      focusPC.close();
      focusPC = null;
    }
    if (mosaicPC) {
      mosaicPC.close();
    }

    const pc = new RTCPeerConnection();
    mosaicPC = pc;

    pc.addTransceiver("video", { direction: "recvonly" });

    pc.ontrack = (event) => {
      const track = event.track;
      track.onunmute = () => {
        const settings = track.getSettings();
        setVideoDimensions(settings)
      };
      attachVideoTrack(mosaicVideo, event.streams[0]);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const res = await safeFetch("/signal", {
      credentials: "include",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "mosaic",
        sdp: offer.sdp,
        type: offer.type
      })
    });

    const answer = await res.json();
    await pc.setRemoteDescription(answer);

  }

  async function startFocusedCamera(name) {
    mosaicTitle = name;

    log("Starting focused camera:", name);
    if (mosaicPC) {
      mosaicPC.close();
      mosaicPC = null;
    }
    if (focusPC) {
      focusPC.close();
    }

    const pc = new RTCPeerConnection();
    focusPC = pc;

    pc.addTransceiver("video", { direction: "recvonly" });

    pc.ontrack = (event) => {
      const track = event.track;
      track.onunmute = () => {
        const settings = track.getSettings();
        setVideoDimensions(settings);
      };
      attachVideoTrack(mosaicVideo, event.streams[0]);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const res = await safeFetch("/signal", {
      credentials: "include",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "focus",
        name,
        sdp: offer.sdp,
        type: offer.type
      })
    });

    const answer = await res.json();
    await pc.setRemoteDescription(answer);

  }

  function handleMosaicClick(event) {
    if (!mosaicVideo) return;

    // If already focused → return to mosaic
    if (isFocusMode) {
      isFocusMode = false;
      currentCamera = null;
      startMosaic();
      return;
    }

    // Determine which tile was clicked
    const rect = mosaicVideo.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const cols = mosaicCols;
    const rows = mosaicRows;

    const cellWidth = rect.width / cols;
    const cellHeight = rect.height / rows;

    const col = Math.floor(x / cellWidth);
    const row = Math.floor(y / cellHeight);

    const index = row * cols + col;

    if (index < 0 || index >= cameras.length) {
      return;
    }

    const cameraName = cameras[index].name;

    isFocusMode = true;
    currentCamera = cameraName;
    startFocusedCamera(cameraName);
  }

  onMount(async () => {
    await loadMosaicDimensions();
    await loadCameras();
    
    startMosaic();
    mosaicVideo.onloadedmetadata = () => {
      videoWidth = mosaicVideo.videoWidth;
      videoHeight = mosaicVideo.videoHeight;
    };
  });

  onDestroy(() => {
    mosaicPC?.close();
    focusPC?.close();
  });


  function getCameraStatus(camera) {
    const status = $cameraStatusStore[camera.name];
    if (!status) return [];   // store not ready yet

    let parts = [];
    
    if (status?.state !== "STREAMING_NORMAL") {
      parts.push('Offline');
    }
    else {
      parts.push(`${status?.recording ? 'REC' : "LIVE"}`);
      parts.push(`FPS ${status?.record_fps}/${status?.read_fps}`);
      
      if (status?.night) {
        parts.push("Night");
      }
    }
    const date = new Date(status?.ts * 1000);
    parts.push(`${date.toLocaleTimeString()}`);

    return parts.join(" | ");
  }

  function getCameraClass(camera) {
    const status = $cameraStatusStore[camera.name];
    if (!status) return "offline";

    if (status.recording) {
      log("Camera recording:", camera.name, ":", status);
    }
    if (status.state === "STREAMING_NORMAL") {
      return status.recording ? "recording" : "live";
    } else {
      return "offline";
    }
  }

  function getCameraObjects(camera) {
    const status = $cameraStatusStore[camera.name];
    if (!status || !status.objects_dict) return [];

    return Object.entries(status.objects_dict)
              .map(([label, colors]) => `${label}: ${colors.join(", ")}`)
              .join("; ");
  }
</script>
<h3 class="mosaic-title">{mosaicTitle}</h3>
<div class="mosaic-container">
  <div class="video-wrapper"
       style="aspect-ratio: {videoWidth} / {videoHeight};">
    <video
      id="mosaic"
      bind:this={mosaicVideo}
      on:click={handleMosaicClick}
      autoplay
      playsinline
    ></video>

    <div class="overlay-grid"
     style="
       --cols: {isFocusMode ? 1 : mosaicCols};
       --rows: {isFocusMode ? 1 : mosaicRows};
       grid-template-columns: repeat({isFocusMode ? 1 : mosaicCols}, 1fr);
       grid-template-rows: repeat({isFocusMode ? 1 : mosaicRows}, 1fr);
     ">
      {#if isFocusMode}
        {#each cameras.filter(c => c.name === currentCamera) as camera}
          {#key $cameraStatusStore[camera.name]?.ts}
            <div class="overlay-cell">
              <div class="overlay-text-block">
                <div class="status-text { getCameraClass(camera) }">
                  {getCameraStatus(camera)}
                </div>
                {#if getCameraObjects(camera)?.length > 0}
                  <div class="objects-text">
                    {getCameraObjects(camera)}
                  </div>
                {/if}
              </div>
            </div>
          {/key}
        {/each}
      {:else}
        {#each cameras as camera}
          {#key $cameraStatusStore[camera.name]?.ts}
            <div class="overlay-cell">
              <div class="overlay-text-block">
                <div class="status-text { getCameraClass(camera) }">
                  {getCameraStatus(camera)}
                </div>
                {#if getCameraObjects(camera)?.length > 0}
                  <div class="objects-text">
                    {getCameraObjects(camera)}
                  </div>
                {/if}
              </div>
            </div>
          {/key}
        {/each}
      {/if}
    </div>
  </div>
</div>
<style>
.mosaic-container {
  width: 100%;
  position: relative;
}

.video-wrapper {
  position: relative;
  width: 100%;
  /* height is determined by aspect-ratio */
}

video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: fill; /* IMPORTANT */
  cursor: pointer;
}

.overlay-grid {
  position: absolute;
  inset: 0;
  display: grid;
  pointer-events: none;
  z-index: 10;
  --tile-w: calc(100vw / var(--cols));
  --tile-h: calc(100vh / var(--rows));
}

.overlay-cell {
  position: relative;
}

.overlay-text-block {
  position: absolute;
  top: 0px;
  left: 0px;
  display: flex;
  flex-direction: column;
  pointer-events: none;
}

.status-text,
.objects-text {
  background: rgba(0,0,0,0.55);
  border-radius: 1px;
  white-space: pre;
  color: #fff;

  /* area-based scaling (browser-safe) */
  font-size: calc((var(--tile-w) + var(--tile-h)) * 0.008);

  padding: 1px;
}
@media (max-width: 700px) {
  .status-text,
  .objects-text {
    font-size: calc((var(--tile-w) + var(--tile-h)) * 0.0045); /* larger text */
  }
}

.status-text.recording {
  background: rgba(200, 0, 0, 0.75); /* red */
}
.status-text.live {
  background: rgba(0, 140, 0, 0.75); /* green */
}
.status-text.live {
  background: rgba(0, 140, 0, 0.75); /* green */
}
.status-text.offline {
  background: rgba(100, 100, 100, 0.75); /* gray */
}
</style>
