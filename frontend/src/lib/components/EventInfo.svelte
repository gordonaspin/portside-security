<script>
  import { page } from '$app/stores';

  export let event = null;
  const log = window.mosaic.log;
  const error = window.mosaic.error;

  // Build absolute metadata URL
  $: metadataHref = (() => {
    if (!event?.metadata_url) return null;

    const m = event.metadata_url;

    // If already absolute, return as-is
    if (m.startsWith("http://") || m.startsWith("https://")) {
      return m;
    }

    // Otherwise prepend correct origin
    return `${$page.url.origin}/${m.replace(/^\//, "")}`;
  })();

</script>

<div class="event-info">
  <h3>Event Info</h3>

  {#if event}
    <div><b>Camera:</b> {event.camera}</div>
    <div><b>Start:</b> {event.start_fmt}</div>
    <div><b>End:</b> {event.end_fmt}</div>

    <div>
      <b>Tags:</b>
      {#each Object.entries(event.tags || {}) as [key, values], i}
        {key}({values.join(", ")}){i < Object.entries(event.tags).length - 1 ? ", " : ""}
      {/each}
    </div>
    <div><b>Metadata:</b><a href="{metadataHref}" target="_blank">{metadataHref}</a></div>
  {:else}
    <p>No event selected.</p>
  {/if}
</div>

<style>
  .event-info {
    padding: 1rem;
    border: 1px solid #666;
    background: #111;
    color: #eee;
    border-radius: 4px;
    font-family:monospace;
  }
  .event-info a {
    white-space: normal;
    word-break: break-all;
    overflow-wrap: anywhere;
  }
  
</style>