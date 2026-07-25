import { writable } from "svelte/store";

export const debug = writable(false);

export function log(...a) {
  debug.update(d => {
    if (d) console.log("[PYNVR]", ...a);
    return d;
  });
}

export function error(...a) {
  debug.update(d => {
    if (d) console.error("[PYNVR]", ...a);
    return d;
  });
}
