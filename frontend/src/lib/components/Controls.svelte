<script lang="ts">
  import { onMount } from "svelte";
  const log = window.mosaic.log;
  const error = window.mosaic.error;

  let confidence = 0.5;
  let motion = 0.1;
  let debug = false;
  let cameras = [];
  let cameraDebug = {};

  onMount(async () => {
    confidence = (await (await fetch("/api/settings/confidence", { credentials: "include" })).json()).value;
    motion     = (await (await fetch("/api/settings/motion", { credentials: "include" })).json()).value;
    debug      = (await (await fetch("/api/settings/debug", { credentials: "include" })).json()).value;
    cameras    = await (await fetch("/api/cameras", { credentials: "include" })).json();

    cameras.forEach(cam => {
      cameraDebug[cam.name] = cam.debug;
    });
  });

  async function updateConfidence() {
    await fetch('/api/settings/confidence', {
      credentials: "include",
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: Number(confidence) })
    });
  }

  async function updateMotion() {
    await fetch('/api/settings/motion', {
      credentials: "include",
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: Number(motion) })
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
    <fieldset>

      <!-- Detection Confidence -->
      <div class="control-row">
        <span class="control-label">Detection Confidence</span>

        <input
          class="value-display"
          type="text"
          value={confidence.toFixed(2)}
          readonly
          tabindex="-1"
        />

        <input
          class="range-slider"
          type="range"
          min="0"
          max="1"
          step="0.01"
          bind:value={confidence}
          on:change={updateConfidence}
        />
      </div>

      <!-- Motion -->
      <div class="control-row">
        <span class="control-label">% Pixel Change in Motion</span>

        <input
          class="value-display"
          type="text"
          value={motion.toFixed(2)}
          readonly
          tabindex="-1"
        />

        <input
          class="range-slider"
          type="range"
          min="0"
          max="1"
          step="0.01"
          bind:value={motion}
          on:change={updateMotion}
        />
      </div>

      <!-- Camera Debug -->
      <div class="camera-debug-section">
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
              bind:checked={debug}
              on:change={updateDebug}
            />
            Verbose Logging
          </label>
        </div>
      </div>

      <!-- Verbose -->
      <div class="verbose-row">

      </div>

    </fieldset>
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

  fieldset {
    padding: 1.25rem;
    border: 1px solid #666;
    background: #111;
    border-radius: 4px;
    margin: 0.5rem;
  }

  /* ⭐ CONTROL ROWS — aligned label | textbox | slider */
  .control-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .control-label {
    width: 80px;
    flex: 0 0 80px;
    text-align: right;
    color: #eee;
  }

  .value-display {
    width: 60px;
    flex: 0 0 60px;
    padding: 4px 6px;
    background: #222;
    color: #eee;
    border: 1px solid #555;
    border-radius: 4px;
    text-align: right;
  }

  .range-slider {
    flex: 1 1 auto;
    min-width: 120px;
    accent-color: #4af;
  }

  /* ⭐ CAMERA DEBUG */
  .camera-debug-section {
    margin-top: 1rem;
    padding-top: 0.5rem;
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

  /* ⭐ VERBOSE ROW — aligned flush-left with camera debug */
  .verbose-row {
    margin-top: 1rem;
  }

  .verbose-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #eee;
  }
</style>
