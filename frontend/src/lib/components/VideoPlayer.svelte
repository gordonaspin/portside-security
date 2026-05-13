<script>
  export let event = null;
  const log = window.mosaic.log;
  const error = window.mosaic.error;
</script>

<h3 class="video-player-title">Video Player</h3>
<!--div class="video-player-panel"-->
  <!--div class="video-player-body"-->
    {#if event}
      <video
        key={event.video_url}
        src={event.video_url}
        autoplay
        controls
        playsinline
      ></video>
    {:else}
      <p class="no-video">No video selected.</p>
    {/if}
  <!--/div-->
<!--/div-->

<style>
  /* Match EventInfo + EventLog panel container */
  .video-player-panel {
    background: #111;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    font-family: "Fira Code", "JetBrains Mono", Consolas, monospace;
    color: #eee;
  }

  .video-player-title {
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
