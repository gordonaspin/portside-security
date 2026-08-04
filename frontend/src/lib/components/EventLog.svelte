<script lang="ts">
  import { onMount } from 'svelte'
  import { logStore, pushLogEntry } from "$lib/stores/logs"
  import { safeFetch } from "$lib/network/safeFetch"
  import { currentEvent } from "$lib/stores/playQueue"
  import type { RecordingEvent } from '$lib/stores/events'
  import { tick } from 'svelte'
  import { log } from '$lib/stores/logging'
  import type { components } from '$lib/types/api';

  type LogEntry = components['schemas']['LogEntry'];

  let container;
  let selectedUrl = null;

  onMount(async () => {
    const res = await safeFetch('/api/logs', { credentials: "include" });
    const data = await res.json();

    // pynvr returns oldest → newest
    // We want newest at top, same as SSE
    for (const logEntry of data.log_entries) {
      pushLogEntry(logEntry);
    }
  })

  function handleClick(e) {
    const link = e.target.closest("a");
    if (!link) return;

    log("clicked event object identity:", e, "id:", e.id);

    e.preventDefault();
    const url = link.getAttribute("href");   // <-- sync only
      // Force reactive block to fire
    selectedUrl = null;
    tick().then(() => {
      selectedUrl = url;
    });
  }

  $: if (selectedUrl) {
    loadSelectedEvent(selectedUrl);
  }

  async function loadSelectedEvent(url) {
    await tick(); // ensure DOM stable

    log("Loading event metadata for", url);
    // Fetch metadata for the clicked log entry
    const res = await safeFetch(url, { credentials: "include"});
    const meta: RecordingEvent = await res.json();
    currentEvent.set(null)
    await tick();
    currentEvent.set({ ...meta });

  }

  function fmt(ts) {
    return new Date(ts * 1000).toLocaleTimeString();
  }

</script>

<h3 class="event-log-title">Event Log</h3>

<div class="event-log" bind:this={container} on:click={handleClick}>
  {#each $logStore as log_entry}
    <div class="log-entry log-{log_entry.level}">
      <span class="log-time">{fmt(log_entry.timestamp)}</span>
      <span class="log-{log_entry.level.toLowerCase()}">{log_entry.level.toUpperCase()}</span>
      <span class="log-message">{log_entry.message}</span>

      {#if log_entry.file_path}
        <a class="log-file" href={log_entry.file_path} target="_blank">{log_entry.anchor}</a>
      {/if}
    </div>
  {/each}
</div>


<style>
  .event-log-title {
    margin: 0;
    padding: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: bold;
    color: #eee;
    font-family: inherit;
  }
  .event-log {
    white-space: normal;
    overflow-wrap: break-word;
    word-break: break-word;

    display: flex;
    flex-direction: column;

    /* ⭐ NEW: constrain height + scroll */
    max-height: 300px;      /* adjust to taste */
    overflow-y: auto;
  }
  .log-info {
    font-size: 0.7rem;
    color: #00c853;
  }
  .log-debug {
    font-size: 0.7rem;
    color: #AA0088;
  }
  .log-warning {
    font-size: 0.7rem;
    color: #ffd600;
  }
  .log-error {
    font-size: 0.7rem;
    color: #ff5252;
  }
  .log-recording {
    font-size: 0.7rem;
    color: #17e8ff;
  }
  .log-info a,
  .log-debug a,
  .log-warning a,
  .log-error a,
  .log-recording a {
    font-size: 0.7rem;
    color: #eee;
    text-decoration: underline;
  }
</style>
