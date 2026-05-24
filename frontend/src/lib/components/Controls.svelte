<script lang="ts">
  import { onMount } from "svelte";
  import { debug, log, error } from "$lib/stores/logging";

  let yoloConfidence = { value: 0.5, min: 0.1, max: 0.9, step: 0.01};
  let motionThreshold = { value: 0.4, min: 0.1, max: 0.9, step: 0.01};
  let minMotionConfidence = { value: 0.4, min: 0.1, max: 0.9, step: 0.01};
  let minMotionFrames = { value: 8, min: 5, max: 20, step: 1};
  let minSumBoxArea = { value: 0.7, min: 0.1, max: 1.5, step: 0.05};

  let verboseDebug = false;
  let cameras = [];
  let selectedCamera = null;
  let cameraDebug = {};

  // NEW: slider fade + shimmer state
  let loading = false;

  onMount(async () => {
    cameras = await (await fetch("/api/cameras", { credentials: "include" })).json();

    if (cameras.length > 0) {
      selectedCamera = cameras[0].name;
      await loadCameraSettings(selectedCamera);
    }

    cameras.forEach(cam => {
      cameraDebug[cam.name] = cam.debug;
    });
  });

  // Load settings for a specific camera
  async function loadCameraSettings(camera) {
    loading = true;   // start shimmer + fade-out

    const res = await fetch(`/api/cameras/${camera}/settings`);
    const s = await res.json();

    applyProfileValue(yoloConfidence, s.yolo_confidence_threshold);
    applyProfileValue(motionThreshold, s.motion_threshold);
    applyProfileValue(minMotionConfidence, s.min_motion_confidence);
    applyProfileValue(minMotionFrames, s.min_motion_frames);
    applyProfileValue(minSumBoxArea, s.min_sum_box_area);

    // small delay to let shimmer animate
    setTimeout(() => loading = false, 150);
  }

  function applyProfileValue(target, src) {
    target.value = src.value;
    target.min   = src.min;
    target.max   = src.max;
    target.step  = src.step;
  }

  // Update a setting for the selected camera
  async function update(setting, value) {
    if (!selectedCamera) return;

    await fetch(`/api/cameras/${selectedCamera}/settings/${setting}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value })
    });
  }

  // Handle clicking a pill
  function selectCamera(cam) {
    selectedCamera = cam;
    loadCameraSettings(cam);
  }

  async function updateDebug() {
    await fetch('/api/settings/debug', {
      credentials: "include",
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: Boolean(verboseDebug) })
    });
  }

  async function updateCameraDebug(name) {
    await fetch(`/api/settings/debug/${name}`, {
      credentials: "include",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value: cameraDebug[name] })
    });
  }

  let collapsed = true;
  function toggle() {
    collapsed = !collapsed;
  }
</script>

<div class="controls-wrapper">
  <div class="controls-header" on:click={toggle}>
    <span>Controls</span>
    <button class="toggle-btn">{collapsed ? "▼" : "▲"}</button>
  </div>

  {#if !collapsed}
    <div class="controls-section">
      <h3>Motion Detection</h3>

      <!-- Camera pill row -->
      <div class="pill-row">
        <div class="slider-block camera-pill-block">
          <label>Cameras</label>
          <div class="camera-pill-row">
            {#each cameras as cam}
              <div
                class="camera-pill {selectedCamera === cam.name ? 'active' : ''}"
                on:click={() => selectCamera(cam.name)}
              >
                {cam.name}
              </div>
            {/each}
          </div>
        </div>
      </div>

      <!-- Slider row with fade + shimmer -->
      <div class="slider-row {loading ? 'loading' : 'loaded'}">
        <div class="slider-grid">

          <!-- YOLO confidence -->
          <div class="slider-block shimmer-item">
            <label>YOLO Confidence: {yoloConfidence.value.toFixed(2)}</label>
            <input type="range"
              min={yoloConfidence.min}
              max={yoloConfidence.max}
              step={yoloConfidence.step}
              bind:value={yoloConfidence.value}
              on:change={() => update("yolo_confidence_threshold", yoloConfidence.value)} />
          </div>

          <!-- motion_threshold -->
          <div class="slider-block shimmer-item">
            <label>Motion Threshold: {motionThreshold.value.toFixed(2)}</label>
            <input type="range"
              min={motionThreshold.min}
              max={motionThreshold.max}
              step={motionThreshold.step}
              bind:value={motionThreshold.value}
              on:change={() => update("motion_threshold", motionThreshold.value)} />
          </div>

          <!-- min_motion_confidence -->
          <div class="slider-block shimmer-item">
            <label>Min Motion Confidence: {minMotionConfidence.value.toFixed(2)}</label>
            <input type="range"
              min={minMotionConfidence.min}
              max={minMotionConfidence.max}
              step={minMotionConfidence.step}
              bind:value={minMotionConfidence.value}
              on:change={() => update("min_motion_confidence", minMotionConfidence.value)} />
          </div>

          <!-- min_motion_frames -->
          <div class="slider-block shimmer-item">
            <label>Min Motion Frames: {minMotionFrames.value}</label>
            <input type="range"
              min={minMotionFrames.min}
              max={minMotionFrames.max}
              step={minMotionFrames.step}
              bind:value={minMotionFrames.value}
              on:change={() => update("min_motion_frames", minMotionFrames.value)} />
          </div>

          <!-- min_sum_box_area -->
          <div class="slider-block shimmer-item">
            <label>Min Sum Box Area: {minSumBoxArea.value.toFixed(2)}</label>
            <input type="range"
              min={minSumBoxArea.min}
              max={minSumBoxArea.max}
              step={minSumBoxArea.step}
              bind:value={minSumBoxArea.value}
              on:change={() => update("min_sum_box_area", minSumBoxArea.value)} />
          </div>

        </div>
      </div>
    </div>

    <!-- Camera Debug -->
    <div class="controls-section">
      <h3>Camera Debug and Logging</h3>

      <div class="camera-debug-row">
        {#each cameras as cam}
          <label class="camera-debug-item">
            <input
              type="checkbox"
              bind:checked={cameraDebug[cam.name]}
              on:change={() => updateCameraDebug(cam.name)}
            />
            {cam.name}
          </label>
        {/each}
        <label class="verbose-label">
          <input
            type="checkbox"
            bind:checked={verboseDebug}
            on:change={() => updateDebug()}
          />
          Verbose Logging
        </label>
      </div>
    </div>
  {/if}
</div>

<style>
/* ============================================================
   PANEL WRAPPER + HEADER
   ============================================================ */
.controls-wrapper {
  border: 1px solid #666;
  background: #111;
  border-radius: 4px;
  font-family: "Fira Code", "JetBrains Mono", Consolas, monospace;
  font-size: 0.7rem;
  color: #eee;
  position: relative;
  z-index: 10000;
  width: 100%;
}

.controls-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.6rem 0.8rem;   /* <-- alignment baseline */
  cursor: pointer;
  user-select: none;
  font-weight: bold;
}

.toggle-btn {
  background: #222;
  color: #eee;
  border: 1px solid #555;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
}

/* ============================================================
   SECTION ALIGNMENT — EVERYTHING LINES UP WITH HEADER
   ============================================================ */
.controls-section {
  padding-left: 0.8rem;   /* <-- matches header left padding */
  padding-right: 0.8rem;
  padding-top: 0.6rem;
  margin-top: 1rem;
  border-top: 1px solid #444;
}

.pill-row,
.slider-row,
.camera-debug-row,
.slider-grid,
.slider-block,
.camera-pill-block,
.verbose-label,
.camera-debug-item {
  margin-left: 0;
  padding-left: 0;
}

/* ============================================================
   CAMERA PILL SELECTOR
   ============================================================ */
.camera-pill-block {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.camera-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

/* Base pill styling */
.camera-pill {
  padding: 0.35rem 0.75rem;
  border-radius: 20px;
  background: #222;
  border: 1px solid #444;
  color: #ccc;
  cursor: pointer;
  user-select: none;
  font-size: 0.85rem;
  transition:
    background 0.15s ease,
    border-color 0.15s ease,
    color 0.15s ease,
    box-shadow 0.15s ease,
    transform 0.15s ease;
}

/* Hover */
.camera-pill:hover {
  background: #333;
  border-color: #666;
}

/* Selected pill */
.camera-pill.active {
  background: #0a84ff;
  border-color: #0a84ff;
  color: white;
  font-weight: 600;
  box-shadow: 0 0 6px rgba(10, 132, 255, 0.5);
  transform: scale(1.05);
}

/* Press animation */
.camera-pill:active {
  transform: scale(0.95);
}

/* ============================================================
   SLIDER ROW — FADE IN + SHIMMER
   ============================================================ */
.slider-row {
  width: 100%;
  opacity: 1;
  transition: opacity 0.25s ease;
}

.slider-row.loading {
  opacity: 0.25;
}

/* Shimmer effect */
.slider-row.loading .shimmer-item {
  position: relative;
  overflow: hidden;
}

.slider-row.loading .shimmer-item::after {
  content: "";
  position: absolute;
  top: 0; left: -150%;
  width: 150%;
  height: 100%;
  background: linear-gradient(
    90deg,
    rgba(255,255,255,0) 0%,
    rgba(255,255,255,0.15) 50%,
    rgba(255,255,255,0) 100%
  );
  animation: shimmer 0.9s infinite;
}

@keyframes shimmer {
  0% { left: -150%; }
  100% { left: 150%; }
}

/* ============================================================
   SLIDER GRID + SLIDER BLOCKS
   ============================================================ */
.slider-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 12px;
}

.slider-block {
  display: flex;
  flex-direction: column;
  min-width: 150px;
}

.slider-block label {
  margin-bottom: 4px;
  color: #ddd;
}

input[type="range"] {
  width: 100%;
}

/* ============================================================
   CAMERA DEBUG ROW
   ============================================================ */
.camera-debug-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 1rem;   /* <-- adds padding below the checkboxes */
}

.camera-debug-item {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  white-space: nowrap;
}

.verbose-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #eee;
}
</style>
