// src/lib/stores/playQueue.ts
import { writable } from "svelte/store";

export const playQueue = writable([]);   // array of event objects
export const isPlaying = writable(false);
export const currentEvent = writable(null);

export function enqueueAuto(ev) {
  playQueue.update(q => [...q, ev]);   // SSE events → back
}

export function enqueueUser(ev) {
  playQueue.update(q => [ev, ...q]);   // user events → front
}