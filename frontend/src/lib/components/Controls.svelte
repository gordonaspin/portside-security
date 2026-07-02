<script lang="ts">
  import { onMount } from "svelte";
  import { safeFetch } from "$lib/network/safeFetch";
  import { log } from "$lib/stores/logging";

  let cameras = [];
  let selectedCamera = null;

  // All settings live here
  let settings = {};
  let cameraDebug = {};
  let verboseDebug = false;

  let loading = false;

  // -----------------------------
  // Load camera list + first camera
  // -----------------------------
  onMount(async () => {
    cameras = await (await safeFetch("/api/cameras", { credentials: "include" })).json();

    cameras.forEach(c => cameraDebug[c.name] = c.debug);

    if (cameras.length > 0) {
      selectedCamera = cameras[0].name;
      await loadCameraSettings(selectedCamera);
    }
  });

  // -----------------------------
  // Load settings for a camera
  // -----------------------------
  async function loadCameraSettings(name) {
    loading = true;

    const res = await safeFetch(`/api/cameras/${name}/settings`);
    const s = await res.json();

    // Replace entire settings object → guaranteed reactivity
    settings = { ...s };

    setTimeout(() => loading = false, 150);
  }

  // -----------------------------
  // Update a single setting
  // -----------------------------
  async function updateSetting(key, value) {
    if (!selectedCamera) return;

    // Update local state (reactive)
    settings = {
      ...settings,
      [key]: {
        ...settings[key],
        value
      }
    };

    // Send to backend
    await safeFetch(`/api/cameras/${selectedCamera}/settings/${key}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value })
    });
  }

  // -----------------------------
  // Switch camera
  // -----------------------------
  function selectCamera(name) {
    selectedCamera = name;
    loadCameraSettings(name);
  }

  // -----------------------------
  // Debug toggles
  // -----------------------------
  async function updateDebug() {
    await safeFetch("/api/settings/debug", {
      credentials: "include",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value: Boolean(verboseDebug) })
    });
  }

  async function updateCameraDebug(name) {
    await safeFetch(`/api/settings/debug/${name}`, {
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


<div class="controls-wrapper {collapsed ? '' : 'expanded'}">
  <div class="controls-header" on:click={toggle}>
    <span>Controls</span>
    <label for="toggle_button"/>
    <button id="toggle_button" class="toggle-btn">{collapsed ? "▼" : "▲"}</button>
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
            <label for="yolo_confidence">YOLO Confidence: {settings.yolo_confidence.value.toFixed(2)}</label>
            <input type="range"
              id="yolo_confidence"
              min={settings.yolo_confidence.min}
              max={settings.yolo_confidence.max}
              step={settings.yolo_confidence.step}
              bind:value={settings.yolo_confidence.value}
              on:change={() => updateSetting("yolo_confidence", settings.yolo_confidence.value)} />
          </div>

          <!-- track_threshold -->
          <div class="slider-block shimmer-item">
            <label for="track_threshold">Track Threshold: {settings.track_threshold.value.toFixed(2)}</label>
            <input type="range"
              id="track_threshold"
              min={settings.track_threshold.min}
              max={settings.track_threshold.max}
              step={settings.track_threshold.step}
              bind:value={settings.track_threshold.value}
              on:change={() => updateSetting("track_threshold", settings.track_threshold.value)} />
          </div>

          <!-- match_threshold -->
          <div class="slider-block shimmer-item">
            <label for="match_threshold">Match Threshold: {settings.match_threshold.value.toFixed(2)}</label>
            <input type="range"
              id="match_threshold"
              min={settings.match_threshold.min}
              max={settings.match_threshold.max}
              step={settings.match_threshold.step}
              bind:value={settings.match_threshold.value}
              on:change={() => updateSetting("match_threshold", settings.match_threshold.value)} />
          </div>

          <!-- track_buffer -->
          <div class="slider-block shimmer-item">
            <label for="track_buffer">Track Buffer: {settings.track_buffer.value}</label>
            <input type="range"
              id="track_buffer"
              min={settings.track_buffer.min}
              max={settings.track_buffer.max}
              step={settings.track_buffer.step}
              bind:value={settings.track_buffer.value}
              on:change={() => updateSetting("track_buffer", settings.track_buffer.value)} />
          </div>

          <!-- minimum_relative_motion -->
          <div class="slider-block shimmer-item">
            <label for="minimum_relative_motion">Min Relative Motion: {settings.minimum_relative_motion.value}</label>
            <input type="range"
              id="minimum_relative_motion"
              min={settings.minimum_relative_motion.min}
              max={settings.minimum_relative_motion.max}
              step={settings.minimum_relative_motion.step}
              bind:value={settings.minimum_relative_motion.value}
              on:change={() => updateSetting("minimum_relative_motion", settings.minimum_relative_motion.value)} />
          </div>

        </div>
      </div>
    </div>

    <!-- Camera Debug -->
    <div class="controls-section">
      <h3>Camera Debug and Logging</h3>

      <div class="camera-debug-row">
        {#each cameras as cam}
          <label for="camera_debug_{cam.name}" class="camera-debug-item">
            <input
              id="camera_debug_{cam.name}"
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
  position: absolute;
  top: 100%;        /* directly below the header */
  right: 0;         /* align to the right edge */
  z-index: 10000;
  width: 30vw;
}
/* Expanded overlay state */
.controls-wrapper.expanded {
  position: absolute;
  top: 100%;        /* directly below the header */
  right: 0;         /* align to the right edge */
  width: 50vw;     /* or whatever width you want */
  max-height: 80vh; /* prevent it from going off-screen */
  overflow-y: auto;
  z-index: 9999;
  box-shadow: 0 4px 12px rgba(0,0,0,0.6);
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
