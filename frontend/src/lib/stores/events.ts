// src/lib/stores/events.ts
import { writable } from "svelte/store";

export const eventStore = writable([]);

const MAX_AGE_SECONDS = 7 * 24 * 60 * 60; // 7 days

export function addEvent(rec) {
  // Encode URLs once, consistently
  const media_url_encoded = rec.media_filename
    .split("/")
    .map(encodeURIComponent)
    .join("/");

  const metadata_url_encoded = rec.metadata_filename
    .split("/")
    .map(encodeURIComponent)
    .join("/");

  const entry = {
    ...rec,
    media_url: "/" + media_url_encoded,
    metadata_url: "/" + metadata_url_encoded
  };

  const now = Math.floor(Date.now() / 1000);

  eventStore.update((list) => {
    // Trim older than 7 days
    const fresh = list.filter(
      (e) => now - e.start_time <= MAX_AGE_SECONDS
    );

    // Append newest at the end (oldest → newest)
    return [...fresh, entry];
  });
}
