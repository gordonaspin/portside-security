// src/hooks.client.ts
import { serverOffline } from '$lib/stores/connection';

export function handleError({ error, event }) {
  console.error("Client error:", error);
}

export async function handleFetch({ request, fetch }) {
  try {
    const res = await fetch(request);

    // If server responds with an error (500, 503, etc)
    if (!res.ok) {
      serverOffline.set(true);
    }

    return res;
  } catch (err) {
    // Network unreachable, server down, DNS failure, etc
    serverOffline.set(true);
    throw err;
  }
}
