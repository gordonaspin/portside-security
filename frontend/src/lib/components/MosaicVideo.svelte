<script>
  import { onMount, onDestroy } from "svelte";
  import { debug, log, error } from "$lib/stores/logging";

  let cameras = [];
  let isFocusMode = false;
  let currentCamera = null;

  let mosaicPC = null;
  let focusPC = null;

  let mosaicVideo; // bind:this
  let mosaicTitle = "";
  let mosaicRows = 0
  let mosaicCols = 0

  async function loadCameras() {
    try {
      const res = await fetch("/api/cameras", { credentials: "include" });
      cameras = await res.json();
    } catch (err) {
      error("Failed to load cameras:", err);
    }
  }

  async function loadMosaicDimensions() {
    try {
      const res = await fetch("/api/mosaic_dimensions", { credentials: "include" });
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
      attachVideoTrack(mosaicVideo, event.streams[0]);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const res = await fetch("/signal", {
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
      attachVideoTrack(mosaicVideo, event.streams[0]);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const res = await fetch("/signal", {
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
    startMosaic();
  });

  onDestroy(() => {
    mosaicPC?.close();
    focusPC?.close();
  });
</script>
<h3 class="mosaic-title">{mosaicTitle}</h3>
<video
  id="mosaic"
  bind:this={mosaicVideo}
  on:click={handleMosaicClick}
  autoplay
  playsinline
  style="width: 100%; height: auto;"
></video>

<style>
  video {
    cursor: pointer;
  }
  .mosaic-title {
    margin: 0;
    padding: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: bold;
    color: #eee;
    font-family: inherit;
  }
</style>
<!--
<script lang="ts">
  export let visible: boolean = true;
  const log = window.mosaic.log;
  const error = window.mosaic.error;

  let mosaicTitle;

  export function setMosaicTitle(t) {
    mosaicTitle = t;
  }

  // expose to mosaic.js
  window.setMosaicTitle = setMosaicTitle;
</script>

<h3 class="mosaic-title">{mosaicTitle}</h3>
<div style="text-align:center; width:100%; margin-bottom:1rem; display:{visible ? 'block' : 'none'};">
  <video
    id="mosaic"
    autoplay
    playsinline
    style="
      width: 100%;
      max-width: 100%;
      border: 2px solid #888;
      background: black;
    "
  ></video>
</div>
<style>
  .mosaic-title {
    margin: 0;
    padding: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: bold;
    color: #eee;
    font-family: inherit;
  }
</style>
-->