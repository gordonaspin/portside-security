import { writable } from "svelte/store";
import type { RecordingEvent } from "./events";

export const playQueue = writable<RecordingEvent[]>([]);
export const currentEvent: RecordingEvent = writable(null);

export function enqueueAuto(ev: RecordingEvent) {
  playQueue.update(q => [...q, ev]);   // SSE events → back
}