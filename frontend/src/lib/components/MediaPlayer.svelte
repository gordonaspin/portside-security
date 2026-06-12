<script lang="ts">
  import { playQueue, isPlaying, currentEvent } from "$lib/stores/playQueue";
  import { log } from "$lib/stores/logging";
  import { tick } from "svelte";

  let videoEl;

  // Reactive trigger
  $: if (!$isPlaying && $playQueue.length > 0) {
    startNextVideo();   // call async function (allowed)
  }

  async function startNextVideo() {
    const ev = $playQueue[0];
    if (!ev) return;

    // Set the new current event
    currentEvent.set(ev);

    // Remove it from the queue
    playQueue.update(q => q.slice(1));

    // Mark as playing
    isPlaying.set(true);

    // Wait for <video> to mount
    await tick();

    // Load video
    if (videoEl && ev.media_filename) {
      log("video playing ", ev.media_filename);
      videoEl.src = ev.media_filename;
      videoEl.play();
    }
  }

  function onEnded() {
    log("video ended");
    isPlaying.set(false);
    currentEvent.set(null);
  }
</script>

<h3 class="media-player-title">Media Player</h3>

{#if $currentEvent}
  <video
    bind:this={videoEl}
    on:ended={onEnded}
    autoplay
    controls
    playsinline
  >
    <track label="English" kind="captions" srclang="en" src="silent.vtt" default>
  </video>
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
  }

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
