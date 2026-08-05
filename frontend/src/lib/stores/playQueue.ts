import { writable, type Writable } from "svelte/store";
import type { components } from '$lib/types/api';

type RecordingEvent = components['schemas']['RecordingEvent'];

export const playQueue = writable<RecordingEvent[]>([]);
export const currentEvent: Writable<RecordingEvent | null> = writable(null);

export function enqueueAuto(ev: RecordingEvent) {
  playQueue.update(q => [...q, ev]);   // SSE events → back
}