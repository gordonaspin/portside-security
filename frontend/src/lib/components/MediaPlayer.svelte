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
      videoEl.src = ev.media_filename;
      videoEl.play();
    }
  }

  function onEnded() {
    isPlaying.set(false);
    currentEvent.set(null);
  }

  // Skip button: jump to the most recent queued event
  function skipToLatest() {
    const q = $playQueue;

    if (q.length === 0) {
      // nothing to skip to
      currentEvent.set(null);
      isPlaying.set(false);
      return;
    }

    // Take the LAST event in the queue
    const latest = q[q.length - 1];

    // Clear queue
    playQueue.set([]);

    // Force playback of the latest event
    currentEvent.set(latest);
    isPlaying.set(true);

    // Load video after DOM updates
    tick().then(() => {
      if (videoEl && latest.media_filename) {
        videoEl.src = latest.media_filename;
        videoEl.play();
      }
    });
  }
</script>

<h3 class="media-player-title">
  Media Player
  <button class="skip-btn" on:click={skipToLatest}>⏩ Skip to Latest</button>
</h3>

{#if $currentEvent}
  <video
    bind:this={videoEl}
    on:ended={onEnded}
    muted
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
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .skip-btn {
    background: #444;
    color: #eee;
    border: 1px solid #666;
    padding: 2px 8px;
    font-size: 0.8rem;
    border-radius: 4px;
    cursor: pointer;
  }

  .skip-btn:hover {
    background: #666;
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
