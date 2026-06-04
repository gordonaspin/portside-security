<script lang="ts">
  import Controls from '$lib/components/Controls.svelte';
  import MosaicVideo from '$lib/components/MosaicVideo.svelte';
  import Timeline from '$lib/components/Timeline.svelte';
  import MediaPlayer from '$lib/components/MediaPlayer.svelte';
  import EventInfo from '$lib/components/EventInfo.svelte';
  import EventLog from "$lib/components/EventLog.svelte";
  import { debug, log, error } from "$lib/stores/logging";
  import { onMount } from 'svelte';

  let logs = []
  let logHtml = '';
  let loadingEvent = false;
  let system_name = ""
  let selectedEvent = null;

  onMount(async () => {
      const res = await fetch('/api/system_name', { credentials: "include" });
      const data = await res.json();
      system_name = data.system_name;
    }
  )

  async function fetchLogs() {
    if (loadingEvent) return;
    loadingEvent = true;

    // newest log is at the END (ascending timestamps)
    const newestTimestamp = logs.length > 0
      ? logs[logs.length - 1].timestamp
      : null;

    const url = newestTimestamp
      ? `/api/logs?since=${newestTimestamp}`
      : `/api/logs`;

    const res = await fetch(url, { credentials: "include" });
    const data = await res.json();

    // append new logs
    logs = [...logs, ...data.logs];

    // rebuild markup
    logHtml = [...logs] // copy
      .reverse()        // newest to oldest  
      .map((log) => {
        const date = new Date(log.timestamp * 1000);
        const formatted = date.toISOString().replace("T", " ").slice(0, 19);

        return `
          <div class="log-entry log-${log.level}">
            <span class="log-time">${formatted}</span>
            <span class="log-level">${log.level.toUpperCase()}</span>
            <span class="log-message">${log.message}</span>
            ${
              log.file_path
                ? `<a class="log-file" href="${log.file_path}" target="_blank">${log.anchor}</a>`
                : ""
            }
          </div>
        `;
      })
      .join("");

    loadingEvent = false;
  }

  setInterval(fetchLogs, 1000);

  function handleSelectEvent(e) {
    selectedEvent = e;
    log("Selected video: ", selectedEvent)
  }

  async function handleLogMedia(metadata_url) {
    loadingEvent = true;
    // Remove leading slash
    const clean = metadata_url.startsWith("/") ? metadata_url.slice(1) : metadata_url;

    const res = await fetch(metadata_url, { credentials: "include" });
    const data = await res.json();
  
    selectedEvent = {
      ...data,
      media_url: "/" + data.media_filename.split("/").map(encodeURIComponent).join("/"),
      metadata_url: "/" + data.metadata_filename.split("/").map(encodeURIComponent).join("/"),
    };
    log("Selected video: ", selectedEvent)
    loadingEvent = false;
  }

</script>
<div class="page">
  <div class="header-row">
    <h3>{system_name}</h3>

    <div class="controls-container">
      <Controls />
    </div>
  </div>

  <div class="panel">
    <MosaicVideo />
  </div>

  <!-- FLEX ROW THAT BECOMES A COLUMN ON MOBILE -->
  <div class="timeline-player-row">
    <div class="panel timeline-panel">
      <Timeline onSelectEvent={handleSelectEvent} />
    </div>

    <div class="panel player-panel">
      <MediaPlayer event={selectedEvent} />
    </div>
  </div>

  <div class="log-info-row">
    <div class="panel log-panel">
      <div class="event-log-content">
        <EventLog html={logHtml} on:selectMedia={(e) => handleLogMedia(e.detail)}/>
      </div>
    </div>

    <div class="panel info-panel">
      <EventInfo event={selectedEvent} />
    </div>
  </div>

</div>


<style>
  /* Page background */
  :global(body) {
    background: #111;
    color: #eee;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    margin: 0;
    padding: 0;
  }

  /* Main page container */
  .page {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
    background: #111;
    min-height: 100vh;
    box-sizing: border-box;
  }

  /* Panels */
  .panel {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 1rem;
    color: #ddd;
  }
  .header-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between; /* title left, controls right */
    width: 100%;
  }
  .controls-container {
    position: relative; /* anchor for absolute overlay */
  }
  .controls-panel {
    flex-shrink: 0; /* prevent shrinking */
    margin-left: 1rem;
  }
  /* ============================================================
    SHARED FLEX ROW BEHAVIOR (Timeline+Player and Log+Info rows)
    ============================================================ */
  .timeline-player-row,
  .log-info-row {
    display: flex;
    flex-direction: row;
    gap: 12px;
    width: 100%;
    align-items: stretch; /* forces equal height */
  }

  /* ============================================================
    LEFT PANELS (Timeline + EventLog)
    ============================================================ */
  .timeline-panel,
  .log-panel {
    flex: 1; /* same width */
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* EventLog panel (left side) */
  .log-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;

    /* ~10 lines tall */
    line-height: 0.9rem;
    max-height: calc(0.9rem * 10 + 1rem); /* padding allowance */
  }

  /* The actual scrollable content */
  .log-panel :global(.event-log-content) {
    flex: 1;              /* fill the panel height */
    overflow-y: auto;     /* scroll only here */
  }
  /* ============================================================
    RIGHT PANELS (MediaPlayer + EventInfo)
    ============================================================ */
  .player-panel,
  .info-panel {
    width: 704px; /* same width */
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
  }

  /* MediaPlayer should fill its panel */
  .player-panel :global(video),
  .player-panel :global(.media-player-root) {
    height: 100%;
    width: 100%;
  }

  /* EventInfo should NOT stretch children */
  .info-panel {
    display: flex;
    flex-direction: column;
  }

  .info-panel :global(*) {
    height: auto;
    width: auto;
  }


  /* ============================================================
    MOBILE LAYOUT
    ============================================================ */
  @media (max-width: 900px) {
    .timeline-player-row,
    .log-info-row {
      flex-direction: column;
    }

    .player-panel,
    .info-panel {
      width: 100%;
    }
  }
</style>
