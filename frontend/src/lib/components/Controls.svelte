<script lang="ts">
  import { onMount } from "svelte";
  import { safeFetch } from "$lib/network/safeFetch";
  import { debug, log } from "$lib/stores/logging";
  import type { components } from '$lib/types/api';

  type Camera = components['schemas']['CameraResponse'];
  type CameraSettings = components['schemas']['CameraSettingsResponse'];
  type SettingValue = components['schemas']['SettingValue'];
  type ClassToggle = components['schemas']['ClassToggle'];

  let cameras: Camera[] = [];
  let selectedCamera: Camera | null = null;

  // All settings live here
  let selectedCameraSettings: CameraSettings = null;
  let loading = false;

  // -----------------------------
  // Load camera list + first camera
  // -----------------------------
  onMount(async () => {
    cameras = await (await safeFetch("/api/cameras", { credentials: "include" })).json();

    if (cameras.length > 0) {
      selectedCamera = cameras[0];
      await loadCameraSettings(selectedCamera);
    }

    await loadVerboseDebug();
  });

  async function loadVerboseDebug() {
    const res = await safeFetch("/api/settings/debug", { credentials: "include" });
    const data = await res.json();

    console.log("[PYNVR] Loaded verbose debug setting:", data.value);
    debug.set(data.value);
  }
  // -----------------------------
  // Load settings for a camera
  // -----------------------------
  async function loadCameraSettings(camera: Camera) {
    loading = true;
    log("Loading settings for camera", camera.name);
    const res = await safeFetch(`/api/cameras/${camera.name}/settings`);
    const s = await res.json();

    // Replace entire settings object → guaranteed reactivity
    selectedCameraSettings = { ...s };

    setTimeout(() => loading = false, 150);
  }

  // -----------------------------
  // Update a single setting
  // -----------------------------
  async function updateSetting(key, value) {
    if (!selectedCamera) return;

    log("Updating setting for", selectedCamera.name, key, "to", value);
    // Update local state (reactive)
    selectedCameraSettings = {
      ...selectedCameraSettings,
      [key]: {
        ...selectedCameraSettings[key],
        value
      }
    };

    // Send to backend
    const body: SettingValue = { value };

    await safeFetch(`/api/cameras/${selectedCamera.name}/settings/${key}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  }

  async function updateClassToggle(className: string, value: boolean) {
    if (!selectedCamera) return;

    log("Updating class toggle for", selectedCamera.name, className, "to", value);
    selectedCameraSettings = {
      ...selectedCameraSettings,
      classes: {
        ...selectedCameraSettings.classes,
        [className]: value
      }
    };

    const body: ClassToggle = { class_name: className, value };

    await safeFetch(`/api/processor/${selectedCamera.name}/class_toggle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  }

  // -----------------------------
  // Switch camera
  // -----------------------------
  function selectCamera(camera: Camera) {
    selectedCamera = camera;
    loadCameraSettings(camera);
  }

  // -----------------------------
  // Debug toggles
  // -----------------------------
  async function updateDebug(value) {
    console.log("[PYNVR] Updating verbose debug to", value);

    const body: SettingValue = { value };

    await safeFetch("/api/settings/debug", {
      credentials: "include",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  }

  async function updateCameraDebug(camera: Camera) {
    log("Updating camera debug for", camera.name, "to", camera.debug);

    const body: SettingValue = { value: camera.debug };

    await safeFetch(`/api/settings/debug/${camera.name}`, {
      credentials: "include",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
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
            {#each cameras as camera}
              <div
                class="camera-pill {selectedCamera === camera ? 'active' : ''}"
                on:click={() => selectCamera(camera)}
              >
                {camera.name}
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
            <label for="yolo_confidence">YOLO Confidence: {selectedCameraSettings.yolo_confidence.value.toFixed(2)}</label>
            <input type="range"
              id="yolo_confidence"
              min={selectedCameraSettings.yolo_confidence.minimum}
              max={selectedCameraSettings.yolo_confidence.maximum}
              step={selectedCameraSettings.yolo_confidence.step}
              bind:value={selectedCameraSettings.yolo_confidence.value}
              on:change={() => updateSetting("yolo_confidence", selectedCameraSettings.yolo_confidence.value)} />
          </div>

          <!-- track_threshold -->
          <div class="slider-block shimmer-item">
            <label for="track_threshold">Track Threshold: {selectedCameraSettings.track_threshold.value.toFixed(2)}</label>
            <input type="range"
              id="track_threshold"
              min={selectedCameraSettings.track_threshold.minimum}
              max={selectedCameraSettings.track_threshold.maximum}
              step={selectedCameraSettings.track_threshold.step}
              bind:value={selectedCameraSettings.track_threshold.value}
              on:change={() => updateSetting("track_threshold", selectedCameraSettings.track_threshold.value)} />
          </div>

          <!-- match_threshold -->
          <div class="slider-block shimmer-item">
            <label for="match_threshold">Match Threshold: {selectedCameraSettings.match_threshold.value.toFixed(2)}</label>
            <input type="range"
              id="match_threshold"
              min={selectedCameraSettings.match_threshold.minimum}
              max={selectedCameraSettings.match_threshold.maximum}
              step={selectedCameraSettings.match_threshold.step}
              bind:value={selectedCameraSettings.match_threshold.value}
              on:change={() => updateSetting("match_threshold", selectedCameraSettings.match_threshold.value)} />
          </div>

          <!-- track_buffer -->
          <div class="slider-block shimmer-item">
            <label for="track_buffer">Track Buffer: {selectedCameraSettings.track_buffer.value}</label>
            <input type="range"
              id="track_buffer"
              min={selectedCameraSettings.track_buffer.minimum}
              max={selectedCameraSettings.track_buffer.maximum}
              step={selectedCameraSettings.track_buffer.step}
              bind:value={selectedCameraSettings.track_buffer.value}
              on:change={() => updateSetting("track_buffer", selectedCameraSettings.track_buffer.value)} />
          </div>

          <!-- minimum_relative_motion -->
          <div class="slider-block shimmer-item">
            <label for="minimum_relative_motion">Min Relative Motion: {selectedCameraSettings.minimum_relative_motion.value}</label>
            <input type="range"
              id="minimum_relative_motion"
              min={selectedCameraSettings.minimum_relative_motion.minimum}
              max={selectedCameraSettings.minimum_relative_motion.maximum}
              step={selectedCameraSettings.minimum_relative_motion.step}
              bind:value={selectedCameraSettings.minimum_relative_motion.value}
              on:change={() => updateSetting("minimum_relative_motion", selectedCameraSettings.minimum_relative_motion.value)} />
          </div>

        </div>
      </div>
    </div>

    <div class="controls-section">
      <h3>Class Filters</h3>

      <!-- Shimmer wrapper identical to slider-row -->
      <div class="class-toggle-row {loading ? 'loading' : 'loaded'}">

        {#each Object.entries(selectedCameraSettings.classes || {}) as [className, enabled]}
          <label class="class-toggle-item shimmer-item">
            <input
              type="checkbox"
              checked={enabled}
              on:change={(e) => updateClassToggle(className, e.target.checked)}
            />
            {className}
          </label>
        {/each}

      </div>
    </div>
    
    <!-- Camera Debug -->
    <div class="controls-section">
      <h3>Camera Debug and Logging</h3>

      <div class="camera-debug-row">
        {#each cameras as camera}
          <label for="camera_debug_{camera.name}" class="camera-debug-item">
            <input
              id="camera_debug_{camera.name}"
              type="checkbox"
              bind:checked={camera.debug}
              on:change={() => updateCameraDebug(camera)}
            />
            {camera.name}
          </label>
        {/each}
        <label class="verbose-label">
          <input
            type="checkbox"
            bind:checked={$debug}
            on:change={() => updateDebug($debug)}
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

.class-toggle-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 1rem;   /* <-- adds padding below the checkboxes */
}
.class-toggle-item {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  white-space: nowrap;
}

/* ============================================================
   CLASS FILTER SHIMMER (same as slider shimmer)
   ============================================================ */
.class-toggle-row.loading {
  opacity: 0.25;
}

.class-toggle-row.loading .shimmer-item {
  position: relative;
  overflow: hidden;
}

.class-toggle-row.loading .shimmer-item::after {
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

</style>
