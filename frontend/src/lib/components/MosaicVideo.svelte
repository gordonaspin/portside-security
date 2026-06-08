<script>
  import { onMount, onDestroy } from "svelte";
  import { debug, log, error } from "$lib/stores/logging";
  import { safeFetch } from '$lib/network/safeFetch';
  import { cameraStatus } from "$lib/stores/cameraStatus";
  import { startStatusEvents } from "$lib/services/statusEvents";

  let cameras = [];
  let isFocusMode = false;
  let currentCamera = null;

  let mosaicPC = null;
  let focusPC = null;

  let mosaicVideo; // bind:this
  let mosaicTitle = "";
  let mosaicRows = 0
  let mosaicCols = 0
  let videoWidth = 3840;
  let videoHeight = 1046;

  async function loadCameras() {
    try {
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

    } catch (err) {
      error("Failed to load cameras:", err);
    }
  }

  function attachVideoTrack(videoEl, stream) {
    if (!videoEl) return;
    videoEl.srcObject = stream;
    videoEl.muted = true;
    videoEl.play().catch(() => {});
  }

  function setVideoDimensions(settings) {
    videoWidth = settings.width;
    videoHeight = settings.height;
    console.log("settings: ", settings);
  }
  // ------------------------------------------------------
  // MOSAIC STREAM
  // ------------------------------------------------------
  async function startMosaic() {
    log("Starting mosaic…");
    mosaicTitle = "All Cameras";

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
      log("Mosaic track received");
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

    log("Mosaic WebRTC connected");
  }

  // ------------------------------------------------------
  // FOCUSED CAMERA STREAM
  // ------------------------------------------------------
  async function startFocusedCamera(name) {
    log(`Starting focused camera ${name}`);
    mosaicTitle = name;

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
      log("Focused track received");
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
        mode: "focus",
        name,
        sdp: offer.sdp,
        type: offer.type
      })
    });

    const answer = await res.json();
    await pc.setRemoteDescription(answer);

    log(`Focused camera ${name} WebRTC connected`);
  }

  // ------------------------------------------------------
  // CLICK HANDLER
  // ------------------------------------------------------
  function handleMosaicClick(event) {
    if (!mosaicVideo) return;

    // If already focused → return to mosaic
    if (isFocusMode) {
      log("Returning to mosaic mode");
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
      log("Clicked outside camera range");
      return;
    }

    const cameraName = cameras[index].name;
    log("Selected camera:", cameraName);

    isFocusMode = true;
    currentCamera = cameraName;
    startFocusedCamera(cameraName);
  }

  // ------------------------------------------------------
  // LIFECYCLE
  // ------------------------------------------------------
  onMount(async () => {
    log("Mosaic.svelte mounted");
    await loadMosaicDimensions()
    await loadCameras();
    const stopSSE = startStatusEvents();
    
    startMosaic();
    mosaicVideo.onloadedmetadata = () => {
      videoWidth = mosaicVideo.videoWidth;
      videoHeight = mosaicVideo.videoHeight;
      console.log("video metadata:", videoWidth, videoHeight);
    };

    return () => {
      stopSSE();
    };
  });

  onDestroy(() => {
    mosaicPC?.close();
    focusPC?.close();
  });
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
          grid-template-columns: repeat({isFocusMode ? 1 : mosaicCols}, 1fr);
          grid-template-rows: repeat({isFocusMode ? 1 : mosaicRows}, 1fr);
        ">
      {#if isFocusMode}
        {#each cameras.filter(c => c.name === currentCamera) as cam}
          <div class="overlay-cell">
            <div class="overlay-text-block">
              <div class="status-text { $cameraStatus[cam.name]?.recording ? 'recording' : 'live' }">
                <span>{$cameraStatus[cam.name]?.status}</span>
              </div>
              <div class="objects-text">
                <span>{$cameraStatus[cam.name]?.objects}</span>
              </div>
            </div>
          </div>
        {/each}
      {:else}
        {#each cameras as cam}
          <div class="overlay-cell">
            <div class="overlay-text-block">
              <div class="status-text { $cameraStatus[cam.name]?.recording ? 'recording' : 'live' }">
                <span>{$cameraStatus[cam.name]?.status}</span>
              </div>
              <div class="objects-text">
                <span>{$cameraStatus[cam.name]?.objects}</span>
              </div>
            </div>
          </div>
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
}

.overlay-cell {
  position: relative;
}

.overlay-text-block {
  position: absolute;
  top: 4px;
  left: 4px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  pointer-events: none;
}

.status-text,
.objects-text {
  background: rgba(0,0,0,0.55);
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.75rem;
  color: #fff;
  width: fit-content;
  white-space: pre;
}
.status-text.recording {
  background: rgba(200, 0, 0, 0.75); /* red */
}
.status-text.live {
  background: rgba(0, 140, 0, 0.75); /* green */
}
</style>
