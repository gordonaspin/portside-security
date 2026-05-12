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
    dispatch("selectVideo", link.getAttribute("href"));
  }
</script>

<div class="event-log-panel">
  <h3 class="event-log-title">{title}</h3>

  <div class="event-log" on:click={handleClick}>
    {@html html}
  </div>
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
    background: #222;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 0.5rem;

    font-family: inherit;
    font-size: 0.9rem;
    line-height: 1.35;
    color: #eee;

    white-space: normal;
    overflow-wrap: break-word;
    word-break: break-word;

    display: flex;
    flex-direction: column;
    gap: 0.25rem;

    /* ⭐ NEW: constrain height + scroll */
    max-height: 300px;      /* adjust to taste */
    overflow-y: auto;
  }

  .event-log a {
    color: #4af;
    text-decoration: underline;
    cursor: pointer;
  }
</style>
