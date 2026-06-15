<script lang="ts">
  import { playQueue, currentEvent } from "$lib/stores/playQueue";
  import { tick } from "svelte";

  let videoEl;

  // Reactive rule: if no current event, pull from queue
  $: if (!$currentEvent && $playQueue.length > 0) {
    const ev = $playQueue[0];
    playQueue.update(q => q.slice(1));
    currentEvent.set(ev);
  }

  // Reactive rule: whenever currentEvent changes → play it
  $: if ($currentEvent) {
    playRecording($currentEvent);
  }


  // Main playback function
  async function playRecording(ev) {
    if (!ev) return;

    if (!videoEl) await tick();
    if (!videoEl) return;

    // Stop current playback
    videoEl.pause();
    videoEl.src = "";
    videoEl.load();

    await tick();

    // Load and play new video
    videoEl.src = ev.media_filename;

    const p = videoEl.play();
    if (p) {
      p.catch(err => {
        if (err.name === "NotAllowedError") {
          const handler = () => {
            videoEl.play();
            window.removeEventListener("click", handler);
          };
          window.addEventListener("click", handler);
        }
      });
    }
  }

  // When video ends, clear currentEvent
  function onEnded() {
    currentEvent.set(null);
  }

  // Skip button: jump to most recent queued event
  function skipToLatest() {
    const q = $playQueue;
    if (q.length === 0) return;

    const latest = q[q.length - 1];
    playQueue.set([]);
    currentEvent.set(latest);
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
