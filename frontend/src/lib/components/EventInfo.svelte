<script lang="ts">
  import { page } from '$app/stores';
  import { currentEvent } from '$lib/stores/playQueue';
  import { debug, log, error } from "$lib/stores/logging";
  
  let event = null;

  $: event = $currentEvent

  // Build absolute metadata URL
  $: metadataHref = (() => {
    if (!event?.metadata_filename) return null;

    const m = event.metadata_filename;

    // If already absolute, return as-is
    if (m.startsWith("http://") || m.startsWith("https://")) {
      return m;
    }

    // Otherwise prepend correct origin
    return `${$page.url.origin}/${m.replace(/^\//, "")}`;
  })();

</script>

<h3 class="event-info-title">Event Info</h3>
<div class="event-info">
  {#if event}
    <div class="row"><span class="label"><b>Camera:</b></span><span class="value">{event.camera}</span></div>
    <div class="row"><span class="label"><b>Start:</b></span><span class="value">{event.start_fmt}</span></div>
    <div class="row"><span class="label"><b>Stop:</b></span><span class="value">{event.end_fmt}</span></div>
    <div class="row"><span class="label"><b>Duration:</b></span><span class="value">{(event.end_time - event.start_time).toFixed(1)}s</span></div>
    <div class="row">
      <span class="label"><b>Tags:</b></span>
      <span class="value">
      {#each Object.entries(event.tags || {}) as [key, values], i}
        {key}({values.join(", ")}){i < Object.entries(event.tags).length - 1 ? ", " : ""}
      {/each}
      </span>
    </div>
    <div class="row"><span class="label"><b>Recorder:</b></span><span class="value">{event.recorder_type}</span></div>
    <div class="row"><b>Metadata:</b><a href="{metadataHref}" target="_blank">click to open</a></div>
  {:else}
    <p>No event selected.</p>
  {/if}
</div>
<style>
  .event-info {
    margin: 0;
    /*padding: 0.5rem 0;*/
    color: #aaa;
    font-size: 0.9rem;
  }
  .event-info a {
    white-space: normal;
    word-break: break-all;
    overflow-wrap: anywhere;
    color: #eee;
    text-decoration: underline;
  }
  .event-info-title {
    margin: 0;
    padding: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: bold;
    color: #eee;
    font-family: inherit;
  }
  .event-info .row {
    display: grid;
    grid-template-columns: 90px auto; /* adjust width as needed */
  }
  .event-info .label {
    font-weight: bold;
  }
  .event-info .value {
    overflow-wrap: break-word; 
  }
</style>
