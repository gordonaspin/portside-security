import { writable } from "svelte/store";

export const eventStore = writable([]);

export function addEvent(rec) {
  eventStore.update((list) => {
    return [...list, rec];
  });
}
