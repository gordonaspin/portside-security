<script lang="ts">
  import { debug, log, error } from "$lib/stores/logging";

  export let event = null;
</script>

<h3 class="media-player-title">Media Player</h3>
{#if event && event.media_filename.endsWith('.mp4')}
  <video
    key={event.media_filename}
    src={event.media_filename}
    autoplay
    controls
    playsinline
  >
  <track label="English" kind="captions" srclang="en" src="silent.vtt" default>
  </video>
{:else if event && event.media_filename.endsWith('.jpg')}
  <img src={event.media_filename}>
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
