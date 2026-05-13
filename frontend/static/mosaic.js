// ======================================================
// GLOBAL MOSAIC + FOCUS + TIMELINE JS
// Hydration‑safe version for SvelteKit 1.x
// ======================================================
let cameras = [];   // [{ name: "B2Lot", enabled: true }, ...]
let isFocusMode = false;
let currentCamera = null;
let mosaicPC = null;
let focusPC = null;

window.mosaic = window.mosaic || {};
window.mosaic.debug = true;
window.mosaic.log = log
window.mosaic.error = error
function log(...args) {
  if (window.mosaic.debug) {
    console.log("[PYNVR]", ...args);
  }
};
function error(...args) {
  if (window.mosaic.debug) {
    console.error("[PYNVR]", ...args);
  }
};

async function loadCameras() {
  try {
    const res = await fetch("/api/cameras", { credentials: "include" });
    cameras = await res.json();
  } catch (err) {
    error("Failed to load cameras:", err);
  }
}

// Load cameras immediately
loadCameras();

// Attach WebRTC tracks to a <video> element
function attachVideoTrack(videoEl, stream) {
  if (!videoEl) return;
  videoEl.srcObject = stream;
  videoEl.muted = true;
  videoEl.play().catch(() => { });
}

// ------------------------------------------------------
// MOSAIC STREAM
// ------------------------------------------------------
async function startMosaic() {
  log("Starting mosaic…");
  window.setMosaicTitle("All Cameras");

  if (focusPC) {
    focusPC.close();
    focusPC = null;
  }

  if (mosaicPC) {
    mosaicPC.close();
  }

  mosaicPC = new RTCPeerConnection();

  const mosaicVideo = document.getElementById("mosaic");
  if (!mosaicVideo) {
    log("Mosaic video element not found");
    return;
  }

  const pc = new RTCPeerConnection();
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
  log("Starting focused camera:", name);
  window.setMosaicTitle(name);

  if (mosaicPC) {
    mosaicPC.close();
    mosaicPC = null;
  }

  if (focusPC) {
    focusPC.close();
  }

  focusPC = new RTCPeerConnection();

  const focusVideo = document.getElementById("mosaic");
  if (!focusVideo) {
    log("Focused video element not found");
    return;
  }

  const pc = new RTCPeerConnection();
  pc.addTransceiver("video", { direction: "recvonly" });

  pc.ontrack = (event) => {
    log("Focused track received");
    attachVideoTrack(focusVideo, event.streams[0]);
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const res = await fetch("/signal", {
    credentials: "include",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode: "focus",
      name: name,
      sdp: offer.sdp,
      type: offer.type
    })
  });

  const answer = await res.json();
  await pc.setRemoteDescription(answer);

  log("Focused camera WebRTC connected");
}

function handleMosaicClick(event) {
  const mosaicVideo = document.getElementById("mosaic");
  if (!mosaicVideo) return;

  // If already focused → return to mosaic
  if (isFocusMode) {
    log("Returning to mosaic mode");
    isFocusMode = false;
    currentCamera = null;
    startMosaic();
    return;
  }

  // Otherwise: determine which tile was clicked
  const rect = mosaicVideo.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  const cols = 5;
  const rows = 2;

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

  // Switch to focus mode
  isFocusMode = true;
  currentCamera = cameraName;
  startFocusedCamera(cameraName);
}

// ------------------------------------------------------
// MOSAIC CLICK → CAMERA SELECTION
// ------------------------------------------------------
function setupMosaicClick() {
  const mosaic = document.getElementById("mosaic");
  if (!mosaic) return;

  mosaic.addEventListener("click", (e) => {
    handleMosaicClick(e)
  });
}

// ------------------------------------------------------
// HYDRATION‑SAFE STARTUP
// ------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  log("mosaic.js loaded");

  // Wait for SvelteKit hydration to finish
  const wait = setInterval(() => {
    const mosaicVideo = document.getElementById("mosaic");
    if (mosaicVideo) {
      clearInterval(wait);
      log("Mosaic element detected — starting WebRTC");
      startMosaic();
      setupMosaicClick();
    }
  }, 100);
});
