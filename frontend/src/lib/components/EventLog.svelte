<script lang="ts">
  import { onMount } from 'svelte';
  import { createEventDispatcher } from "svelte";
  import { debug, log, error } from "$lib/stores/logging";
  import { logStore, addLog } from "$lib/stores/logs"
  import { safeFetch } from "$lib/network/safeFetch";

  const dispatch = createEventDispatcher();

  let container;

  onMount(async () => {
    const res = await safeFetch('/api/logs', { credentials: "include" });
    const data = await res.json();

    // Initial load: newest at top
    for (const log of data.logs) {
      addLog(log);
    }
  });

  function handleClick(e) {
    const link = e.target.closest("a");
    if (!link) return;

    e.preventDefault();
    dispatch("selectMedia", link.getAttribute("href"));
  }

  function fmt(ts) {
    return new Date(ts * 1000).toLocaleTimeString();
  }
</script>

<h3 class="event-log-title">Event Log</h3>

<div class="event-log" bind:this={container} on:click={handleClick}>
  {#each $logStore as log}
    <div class="log-entry log-{log.level}">
      <span class="log-time">{fmt(log.timestamp)}</span>
      <span class="log-{log.level.toLowerCase()}">{log.level.toUpperCase()}</span>
      <span class="log-camera">{log.camera}</span>
      <span class="log-message">{log.message}</span>

      {#if log.file_path}
        <a class="log-file" href={log.file_path} target="_blank">{log.anchor}</a>
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
  .log-warn {
    font-size: 0.7rem;
    color: #ffd600;
  }
  .log-error {
    font-size: 0.7rem;
    color: #ff5252;
  }
  .log-record {
    font-size: 0.7rem;
    color: #17e8ff;
  }
  .log-info a,
  .log-debug a,
  .log-warn a,
  .log-error a,
  .log-record a {
    font-size: 0.7rem;
    color: #eee;
    text-decoration: underline;
  }
</style>
