// src/lib/stores/events.ts
import { writable } from "svelte/store";

export const eventStore = writable([]);

const MAX_AGE_SECONDS = 24 * 60 * 60; // 7 days

export function addEvent(rec) {
  eventStore.update((list) => {
    return [...list, rec];
  });
}
