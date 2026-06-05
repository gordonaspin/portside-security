import { serverOffline } from '$lib/stores/connection';

export async function safeFetch(url: string, options?: RequestInit) {
  try {
    const res = await fetch(url, options);

    if (!res.ok) {
      serverOffline.set(true);
    }

    return res;
  } catch (err) {
    serverOffline.set(true);
    throw err;
  }
}
