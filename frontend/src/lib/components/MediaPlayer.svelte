<script>
  export let event = null;
  const log = window.mosaic.log;
  const error = window.mosaic.error;
</script>

<h3 class="media-player-title">Media Player</h3>
{#if event && event.media_url.endsWith('.mp4')}
  <video
    key={event.media_url}
    src={event.media_url}
    autoplay
    controls
    playsinline
  ></video>
{:else if event && event.media_url.endsWith('.jpg')}
  <img src={event.media_url}>
{:else}
  <p class="no-video">No video selected.</p>
{/if}

<style>
  .media-player-title {
    margin: 0;
    padding: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: bold;
    color: #eee;
    font-family: inherit;
  }

  /* Video element styling */
  video {
    width: 100%;
    border-radius: 4px;
    background: black;
    border: 1px solid #444;
  }

  .no-video {
    margin: 0;
    padding: 0.5rem 0;
    color: #aaa;
    font-size: 0.9rem;
  }
</style>
