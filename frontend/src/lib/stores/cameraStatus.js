import { writable } from "svelte/store";

export const cameraStatus = writable({});
// shape:
// {
//   [cameraName]: {
//     status: "Recording",
//     objects: "person, car"
//   }
// }
