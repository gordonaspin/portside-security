import { writable } from "svelte/store";
import type { components } from '$lib/types/api';

type CameraStatus = components['schemas']['CameraStatusResponse'];

export const cameraStatusStore = writable<Record<string, CameraStatus>>({});
