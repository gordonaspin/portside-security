import { writable } from "svelte/store";

export interface RecordingEvent {
  camera: string;
  fps: number;

  // e.g. { car: ["gray"], person: ["black", "white"] }
  tags: TagMap;

  media_filename: string;
  log_filename: string;

  start_time: number;       // epoch seconds (float)
  end_time: number;         // epoch seconds (float)
  duration: number;         // seconds (float)

  duration_fmt: string;     // "0:00:14"
  start_fmt: string;        // "2026/06/16 16:44:10"
  end_fmt: string;          // "2026/06/16 16:44:25"

  metadata_filename: string;
  recorder_type: string;
}

export type TagMap = {
  [objectType: string]: string[];   // e.g. "car": ["gray"]
};

export const eventStore = writable<RecordingEvent[]>([]);

export function addEvent(rec) {
  eventStore.update((list) => {
    return [...list, rec];
  });
}
