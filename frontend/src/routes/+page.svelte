<script lang="ts">
  import Controls from '$lib/components/Controls.svelte';
  import MosaicVideo from '$lib/components/MosaicVideo.svelte';
  import Timeline from '$lib/components/Timeline.svelte';
  import VideoPlayer from '$lib/components/VideoPlayer.svelte';
  import EventInfo from '$lib/components/EventInfo.svelte';
  import EventLog from "$lib/components/EventLog.svelte";

  let logHtml = '';
  let loadingEvent = false;

  async function fetchLogs() {
    if (loadingEvent) return;

    const res = await fetch('/api/logs', { credentials: "include" });
    const data = await res.json();
    if (data.html !== logHtml) {
      logHtml = data.html;
    }
  }
  setInterval(fetchLogs, 1000);

  let selectedEvent = null;

  function fmt(ts) {
    const d = new Date(ts * 1000);
    const pad = n => String(n).padStart(2, '0');
    return `${pad(d.getMonth()+1)}/${pad(d.getDate())}/${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }

  function handleSelectEvent(e) {
    selectedEvent = {
      ...e,
      start_fmt: fmt(e.start_time),
      end_fmt: fmt(e.end_time)
    };
    log("Selected video: ", selectedEvent)
  }

  async function handleLogVideo(url) {
    loadingEvent = true;
    // Remove leading slash
    const clean = url.startsWith("/") ? url.slice(1) : url;

    // Split into parts
    const parts = clean.split("/");   // ["recordings", "B67Lot", "FILE.mp4"]
    const camera = parts[1];
    const filename = parts[2];

    // Build metadata path
    const metadataPath =
      `/recordings/metadata/${camera}/${filename.replace(".mp4", ".json")}`;


    // Fetch metadata JSON
    const res = await fetch(metadataPath, { credentials: "include" });
    const data = await res.json();
      console.log("metadata keys:", Object.keys(data));
    const video_url_encoded = data.output.split("/").map(encodeURIComponent).join("/");
    const metadata_url_encoded = data.metadata.split("/").map(encodeURIComponent).join("/");

  selectedEvent = {
    camera: data.camera,
    tags: data.tags,
    output: data.output,
    metadata: data.metadata,
    video_url: "/" + video_url_encoded,
    metadata_url: "/" + metadata_url_encoded,
    start_fmt: fmt(data.start_time),
    end_fmt: fmt(data.end_time)
  };
    loadingEvent = false;
  }

</script>
<div class="page">
  <h3>Portside Security Cameras</h3>
  <div class="panel">
    <Controls />
  </div>
  <div class="panel">
    <MosaicVideo />
  </div>
  <div class="panel">
    <Timeline onSelectEvent={handleSelectEvent} />
  </div>
  <div class="panel">
    <EventInfo event={selectedEvent} />
  </div>
  <div class="panel">
    <VideoPlayer event={selectedEvent} />
  </div>
  <div class="panel">
    <EventLog html={logHtml} on:selectVideo={(e) => handleLogVideo(e.detail)} />
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

  /* Panels (match EventInfo / Controls look) */
  .panel {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 1rem;
    color: #ddd;
  }

  .panel h2,
  .panel h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: #fff;
  }

  /* Buttons (same style as Controls) */
  button {
    background: #222;
    border: 1px solid #444;
    color: #eee;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    font-family: inherit;
    cursor: pointer;
    transition: background 0.15s;
  }

  button:hover {
    background: #333;
  }

  /* Inputs */
  input,
  select {
    background: #222;
    border: 1px solid #444;
    color: #eee;
    padding: 0.4rem;
    border-radius: 4px;
    font-family: inherit;
  }

  /* Responsive layout */
  @media (max-width: 600px) {
    .page {
      padding: 0.5rem;
    }
    .panel {
      padding: 0.75rem;
    }
  }
</style>
