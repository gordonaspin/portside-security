<script lang="ts">
  export let data; // contains { user } from +layout.server.js

  import favicon from '$lib/assets/favicon.svg';
  import { onDestroy } from 'svelte';
  import { cameraStatusStore } from '$lib/stores/cameraStatus';
  import { addEvent } from "$lib/stores/events";
  import { addLog } from "$lib/stores/logs";
  import { serverOffline } from '$lib/stores/connection';
  import { enqueueAuto } from '$lib/stores/playQueue.js';
  import { RecordingEvent } from '$lib/stores/events';
  import { log } from '$lib/stores/logging.js'

  let es: EventSource | null = null;
  let interval: ReturnType<typeof setInterval>;

  $: if (data.user && !es) {
    startEventStream();
  }

  $: if (!data.user && es) {
    es.close();
    es = null;
  }

  function startEventStream() {
    const source = new EventSource('/api/stream', { withCredentials: true });
    es = source;

    source.onopen = () => serverOffline.set(false);

    source.onerror = () => {
      serverOffline.set(true);
      source.close();
      es = null;
    };

    source.addEventListener("cameraStatus", (ev) => {
      const cam = JSON.parse(ev.data).data;
      cameraStatusStore.update((current) => ({ ...current, [cam.name]: cam }));
    });

    source.addEventListener("newEvent", (ev) => {
      const event: RecordingEvent = JSON.parse(ev.data).data;
      addEvent(event);
      enqueueAuto(event)
    });

    source.addEventListener("logLine", (ev) => {
      addLog(JSON.parse(ev.data).data);
    });

  }

  $: if ($serverOffline && !interval) {
    interval = setInterval(async () => {
      try {
        await fetch('/api/server_time');
        location.reload();
      } catch {}
    }, 5000);
  }

  $: if (!$serverOffline && interval) {
    clearInterval(interval);
    interval = null;
  }

  onDestroy(() => {
    es?.close();
    es = null;
  });

</script>

{#if $serverOffline}
  <div class="offline-banner">
    ⚠️ Server unreachable
    <button class="reload-btn" on:click={() => location.reload()}>
      Reload now
    </button>
  </div>
{/if}

<svelte:head>
  <link rel="icon" href={favicon} />
</svelte:head>

<slot />

<style>
.offline-banner {
  background: #ff4444;
  color: white;
  padding: 0.75rem 1rem;
  font-weight: bold;
  text-align: center;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 99999;

  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}

.reload-btn {
  background: white;
  color: #ff4444;
  border: none;
  padding: 0.4rem 0.8rem;
  border-radius: 4px;
  font-weight: bold;
  cursor: pointer;
}

.reload-btn:hover {
  background: #ffe5e5;
}
</style>

