<script lang="ts">
  import { onMount } from "svelte";
  const log = window.mosaic.log;
  const error = window.mosaic.error;

  let yoloConfidence = 0;
  let motionThreshold = 0;
  let minMotionConfidence = 0;
  let minMotionFrames = 0;
  let minSumBoxArea = 0;
  let debug = false;
  let cameras = [];
  let cameraDebug = {};

  onMount(async () => {
    const load = async (val) => {
        const r = await fetch(`/api/settings/${val}`, { credentials: "include" });
        if (r.ok) return (await r.json()).value;
    };

    yoloConfidence      = await load("yolo_confidence");
    motionThreshold     = await load("motion_threshold");
    debug               = await load("debug");
    minMotionFrames     = await load("min_motion_frames");
    minMotionConfidence = await load("min_motion_confidence");
    minSumBoxArea       = await load("min_sum_box_area");

    cameras    = await (await fetch("/api/cameras", { credentials: "include" })).json();

    cameras.forEach(cam => {
      cameraDebug[cam.name] = cam.debug;
    });
  });

  // Update backend when sliders move
  async function update(path, value) {
      await fetch(`/api/settings/${path}`, {
          credentials: "include",
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ value })
      });
  }


  async function updateDebug() {
    await fetch('/api/settings/debug', {
      credentials: "include",
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: Boolean(debug) })
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
      <div class="control-row">
        <div class="slider-grid">
            <!-- YOLO confidence -->
            <div class="slider-block">
                <label>YOLO Confidence: {yoloConfidence.toFixed(2)}</label>
                <input type="range"
                    min="0.1" max="0.9" step="0.01"
                    bind:value={yoloConfidence}
                    on:change={() => update("yolo_confidence", yoloConfidence)} />
            </div>

            <!-- motion_threshold -->
            <div class="slider-block">
                <label>Motion Threshold: {motionThreshold}</label>
                <input type="range"
                    min="0.1" max="0.9" step="0.01"
                    bind:value={motionThreshold}
                    on:change={() => update("motion_threshold", motionThreshold)} />
            </div>

            <!-- motion_confidence_min -->
            <div class="slider-block">
                <label>Min Motion Confidence: {minMotionConfidence.toFixed(2)}</label>
                <input type="range"
                    min="0.1" max="1.0" step="0.01"
                    bind:value={minMotionConfidence}
                    on:change={() => update("min_motion_confidence", minMotionConfidence)} />
            </div>

            <!-- min_motion_frames -->
            <div class="slider-block">
                <label>Min Motion Frames: {minMotionFrames}</label>
                <input type="range"
                    min="1" max="20" step="1"
                    bind:value={minMotionFrames}
                    on:change={() => update("min_motion_frames", minMotionFrames)} />
            </div>

            <!-- min_sum_box_area -->
            <div class="slider-block">
                <label>Min Sum Box Area: {(minSumBoxArea).toFixed(2)}</label>
                <input type="range"
                    min="1000" max="5000" step="100"
                    bind:value={minSumBoxArea}
                    on:change={() => update("min_sum_box_area", minSumBoxArea)} />
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
              on:change={() => update(`debug/${cam.name}`, cameraDebug[cam.name])}
            />
            {cam.name}
          </label>
        {/each}
        <label class="verbose-label">
          <input
            type="checkbox"
            bind:checked={debug}
            on:change={() => update("debug", minSumBoxArea)}
          />
          Verbose Logging
        </label>
      </div>
    </div>
  {/if}
</div>

<style>
  /* Wrapper */
  .controls-wrapper {
    border: 1px solid #666;
    background: #111;
    border-radius: 4px;
    font-family: "Fira Code", "JetBrains Mono", Consolas, monospace;
    font-size: 0.7rem; /* or 14px */
    color: #eee;
    position: relative;
    z-index: 10000;
    width: 100%;
  }

  /* Header */
  .controls-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.8rem;
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

  .control-row {
    display: flex;
    flex-direction: row;
    align-items: center;
  }

  .controls-section {
    padding: 0.6rem 0.8rem;
    margin-top: 1rem;
    border-top: 1px solid #444;
  }

  .camera-debug-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
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
