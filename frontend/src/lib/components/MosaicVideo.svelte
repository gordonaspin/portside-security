<script lang="ts">
  export let visible: boolean = true;
  const log = window.mosaic.log;
  const error = window.mosaic.error;

  let mosaicTitle;

  export function setMosaicTitle(t) {
    mosaicTitle = t;
  }

  // expose to mosaic.js
  window.setMosaicTitle = setMosaicTitle;
</script>

<h3 class="mosaic-title">{mosaicTitle}</h3>
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
<style>
  .mosaic-title {
    margin: 0;
    padding: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: bold;
    color: #eee;
    font-family: inherit;
  }
</style>