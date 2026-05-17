<script>
  import { createEventDispatcher } from "svelte";

  export let html = "";
  export let title = "Event Log";

  const dispatch = createEventDispatcher();

  function handleClick(e) {
    const link = e.target.closest("a");
    console.log("EventLog click:", e.target);
    if (!link) return;

    e.preventDefault();
    dispatch("selectMedia", link.getAttribute("href"));
  }
</script>

<h3 class="event-log-title">{title}</h3>
<div class="event-log" on:click={handleClick}>
  {@html html}
</div>

<style>
  .event-log-panel {
    background: #111;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    font-family: "Fira Code", "JetBrains Mono", Consolas, monospace;
  }

  .event-log-title {
    margin: 0;
    padding: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: bold;
    color: #eee;
    font-family: inherit;
  }

  .event-log {
/*    background: #222;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 0.5rem;

    font-family: inherit;
    font-size: 0.9rem;
    line-height: 1.35;
    color: #eee;
    gap: 0.25rem;
*/
    white-space: normal;
    overflow-wrap: break-word;
    word-break: break-word;

    display: flex;
    flex-direction: column;

    /* ⭐ NEW: constrain height + scroll */
    max-height: 300px;      /* adjust to taste */
    overflow-y: auto;
  }
  :global(.log-info) {
    font-size: 0.7rem;
    color: #00c853;
  }
  :global(.log-debug) {
    font-size: 0.7rem;
    color: #AA0088;
  }
  :global(.log-warn) {
    font-size: 0.7rem;
    color: #ffd600;
  }
  :global(.log-error) {
    font-size: 0.7rem;
    color: #ff5252;
  }
  :global(.log-record) {
    font-size: 0.7rem;
    color: #17e8ff;
  }
  :global(.log-info a),
  :global(.log-debug a),
  :global(.log-warn a),
  :global(.log-error a),
  :global(.log-record a) {
    font-size: 0.7rem;
    color: #eee;
    text-decoration: underline;
  }
</style>
