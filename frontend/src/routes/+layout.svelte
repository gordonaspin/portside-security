<script lang="ts">
  import favicon from '$lib/assets/favicon.svg';
  export const ssr = false;

  import { serverOffline } from '$lib/stores/connection';

  $: offline = $serverOffline;

  // Auto‑reload every 5 seconds while offline
  let interval: ReturnType<typeof setInterval>;

  $: if (offline) {
    // Start interval only once
    if (!interval) {
      interval = setInterval(async () => {
        try {
          await fetch('/api/server_time', { method: 'GET' });
          location.reload();
        } catch {
          // do nothing
        }
      }, 5000);
    }
  } else {
    // Clear interval when back online
    if (interval) {
      clearInterval(interval);
      interval = null;
    }
  }
</script>

{#if offline}
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

