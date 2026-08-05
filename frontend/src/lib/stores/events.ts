import { writable } from "svelte/store";
import type { components } from '$lib/types/api';

type RecordingEvent = components['schemas']['RecordingEvent'];

export const eventStore = writable<RecordingEvent[]>([]);

export function addEvent(rec: RecordingEvent) {
  eventStore.update((list) => {
    return [...list, rec];
  });
}
