<script lang="ts">
  import Controls from '$lib/components/Controls.svelte';
  import MosaicVideo from '$lib/components/MosaicVideo.svelte';
  import Timeline from '$lib/components/Timeline.svelte';
  import MediaPlayer from '$lib/components/MediaPlayer.svelte';
  import EventInfo from '$lib/components/EventInfo.svelte';
  import EventLog from "$lib/components/EventLog.svelte";
  import { debug, log, error } from "$lib/stores/logging";
  import { onMount } from 'svelte';
  import { safeFetch } from '$lib/network/safeFetch';

  let system_name = ""

  onMount(async () => {
      const res = await safeFetch('/api/system_name', { credentials: "include" });
      const data = await res.json();
      system_name = data.system_name;
    }
  )


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
      <Timeline />
    </div>

    <div class="panel player-panel">
      <MediaPlayer />
    </div>
  </div>

  <div class="log-info-row">
    <div class="panel log-panel">
      <div class="event-log-content">
        <EventLog />
      </div>
    </div>

    <div class="panel info-panel">
      <EventInfo />
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
