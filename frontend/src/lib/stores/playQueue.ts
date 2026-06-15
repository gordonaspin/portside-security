import { writable } from "svelte/store";

export const playQueue = writable([]);   // array of event objects
export const currentEvent = writable(null);

export function enqueueAuto(ev) {
  playQueue.update(q => [...q, ev]);   // SSE events → back
}