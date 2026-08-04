import { writable } from 'svelte/store';

export const logStore = writable([]);

const MAX_LOGS = 1000;

export function pushLogEntry(entry) {
  logStore.update((list) => {
    const updated = [entry, ...list];   // newest at top
    return updated.slice(0, MAX_LOGS);  // trim excess
  });
}