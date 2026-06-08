import { cameraStatus } from "$lib/stores/cameraStatus";

export function startStatusEvents() {
  const es = new EventSource("/camera_status", { withCredentials: true });

  es.onmessage = (ev) => {
    const arr = JSON.parse(ev.data); 
    // arr = [{ name, status, objects }, ...]

    cameraStatus.update((s) => {
      const next = { ...s };
      for (const cam of arr) {
        next[cam.name] = {
          status: cam.status,
          objects: cam.objects,
          recording: cam.recording
        };
      }
      return next;
    });
  };

  es.onerror = () => {
    console.warn("SSE disconnected");
  };

  return () => es.close();
}
