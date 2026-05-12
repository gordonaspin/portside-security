<script lang="ts">
  export let visible: boolean = true;
  const log = window.mosaic.log;
  const error = window.mosaic.error;
</script>

<div style="text-align:center; width:100%; margin-bottom:1rem; display:{visible ? 'block' : 'none'};">
  <video
    id="mosaic"
    autoplay
    playsinline
    style="
      width: 100%;
      max-width: 100%;
      border: 2px solid #888;
      background: black;
    "
  ></video>
</div>
