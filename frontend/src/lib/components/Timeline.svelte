<script lang="ts">
  import { onMount, onDestroy, tick } from "svelte";
  import { debug, log, error } from "$lib/stores/logging";
  import { safeFetch } from "$lib/network/safeFetch";
  import { eventStore, addEvent } from "$lib/stores/events";

  export let onSelectEvent = () => {};

  const TICK_HEIGHT = 20;
  const ROW_HEIGHT = 40;
  const HEADER_HEIGHT = TICK_HEIGHT + 36;
  const LEGEND_HEIGHT = 24;
  const HOUR = 3600;
  const isMobile =
    typeof window !== "undefined" &&
    /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  const MIN_ZOOM = 1 / 60;
  const MAX_ZOOM = isMobile ? 4 : 24;

  let cameras = [];
  let classes = [];
  let classColors = {};
  let selectedClasses = new Set();
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null;
  let initialAligned = false;
  let zoomHours = 24;
  let offsetSeconds = 0;
  let computedLeftMargin = 140;
  let serverNow = 0;
  let legendItems = [];
  let selectedEventId: number | null = null;

  let ro: ResizeObserver;
  let serverTimeInterval: number;

  // hover / tooltip
  let hoverEvent = null;
  let mouseX = 0;
  let mouseY = 0;
  let tooltipLeft = 0;
  let tooltipTop = 0;

  // gestures
  let gestureMode = "none";
  let startX = 0;
  let startY = 0;
  let pointers = new Map<number, { x: number; y: number }>();
  let panStartX = 0;
  let panStartOffset = 0;
  let isDragging = false;
  let zoomStartY = 0;
  let zoomStartHours = 0;
  let zoomCenterX = 0;
  let zoomCenterSeconds = 0;
  let tapStart: { x: number; y: number; time: number } | null = null;

  $: drawTimeline(
    $eventStore,
    classes,
    selectedClasses,
    zoomHours,
    offsetSeconds,
    serverNow,
    computedLeftMargin,
    selectedEventId
  );

  onMount(async () => {
    ctx = canvas.getContext("2d");

    await loadServerTime();
    await loadClasses();
    await loadCameras();

    if (isMobile) {
      zoomHours = 4;
      offsetSeconds = 0;
    }
    zoomHours = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoomHours));

    computedLeftMargin = computeDynamicLeftMargin();

    await fetchEventsForCurrentWindow();
    drawTimeline();

    ro = new ResizeObserver(() => {
      computedLeftMargin = computeDynamicLeftMargin();
      drawTimeline();
    });
    ro.observe(canvas);

    serverTimeInterval = window.setInterval(async () => {
      await loadServerTime();
      await fetchEventsForCurrentWindow();
      drawTimeline();
    }, 60000);
  });

  async function loadServerTime() {
    const res = await safeFetch("/api/server_time", { credentials: "include" });
    const data = await res.json();
    serverNow = data.epoch;
  }

  async function loadClasses() {
    const res = await safeFetch("/api/classes", { credentials: "include" });
    const data = await res.json();
    classes = data.classes;

    const palette = [
      "#ff4444",
      "#4488ff",
      "#aa44ff",
      "#ffff44",
      "#ff8844",
      "#aa6633",
      "#44ff44",
      "#44ffff"
    ];

    classColors = {};
    classes.forEach((cls, i) => {
      classColors[cls] = palette[i % palette.length];
    });
  }

  async function loadCameras() {
    const res = await safeFetch("/api/cameras", { credentials: "include" });
    cameras = await res.json();
  }

  async function fetchEventsForCurrentWindow() {
    const { start, end } = getTimelineBounds2();
    const url = `/api/events?start=${start}&end=${end}&mobile=${isMobile ? 1 : 0}`;

    const res = await safeFetch(url, { credentials: "include" });
    const data = await res.json();

    eventStore.set(data.events);
  }

  $: if (!initialAligned && $eventStore.length > 0) {
    if (!isMobile) {
      const latestEnd = Math.max(...$eventStore.map((e) => e.end_time));
      offsetSeconds = Math.max(0, serverNow - latestEnd);
    } else {
      zoomHours = 4;
      offsetSeconds = 0;
    }
    initialAligned = true;
  }

  $: if (initialAligned && $eventStore.length > 0) {
    const latestEnd = Math.max(...$eventStore.map((e) => e.end_time));
    const { end } = getTimelineBounds();
    const atLiveEdge = Math.abs(end - latestEnd) < 1;

    if (atLiveEdge) {
      offsetSeconds = Math.max(0, serverNow - latestEnd);
    }
  }

  function getTimelineBounds() {
    const latestEnd =
      $eventStore.length > 0
        ? Math.max(...$eventStore.map((e) => e.end_time))
        : serverNow;

    const edge = Math.max(serverNow, latestEnd);
    const end = edge - offsetSeconds;
    const start = end - zoomHours * HOUR;

    return { start, end };
  }

  function getTimelineBounds2() {
    const end = serverNow - offsetSeconds;
    const start = end - zoomHours * HOUR;
    return { start, end };
  }

  function xFor(ts: number) {
    const { start, end } = getTimelineBounds();
    const total = end - start;
    return (
      computedLeftMargin +
      ((ts - start) / total) * (canvas.width - computedLeftMargin)
    );
  }

  function drawTimeline() {
    if (!ctx) return;
    const start = Date.now();

    const w = (canvas.width = canvas.clientWidth);
    const h = (canvas.height = canvas.clientHeight);

    ctx.clearRect(0, 0, w, h);

    drawBackground(w, h);
    drawLegend();
    drawTimeTicks(w);
    drawCameraRows(w);
    drawEvents(w);
    if (!hasVisibleEvents()) {
      drawNoEventsMessage();
    }
    log("drawTimeline took ms: ", Date.now() - start);
  }

  function drawBackground(w: number, h: number) {
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, w, h);
  }

  function roundRectPath(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    r: number
  ) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function handleLegendClick(mx: number, my: number) {
    for (const item of legendItems) {
      if (
        mx >= item.x &&
        mx <= item.x + item.w &&
        my >= item.y &&
        my <= item.y + item.h
      ) {
        if (selectedClasses.has(item.cls)) {
          selectedClasses.delete(item.cls);
        } else {
          selectedClasses.add(item.cls);
        }
        selectedClasses = new Set(selectedClasses);
        return true;
      }
    }
    return false;
  }

  function drawLegend() {
    ctx.font = "12px sans-serif";
    ctx.textBaseline = "middle";

    const y = canvas.height - LEGEND_HEIGHT + 12;
    let x = computedLeftMargin;

    legendItems = [];

    classes.forEach((cls) => {
      const textWidth = ctx.measureText(cls).width;
      const paddingX = 8;
      const paddingY = 4;
      const w = textWidth + paddingX * 2;
      const h = 20;
      const r = 6;

      const isSelected = selectedClasses.has(cls);

      const bx = x;
      const by = y - h / 2;

      ctx.fillStyle = isSelected
        ? "rgba(255,255,255,0.36)"
        : "rgba(255,255,255,0.06)";
      roundRectPath(ctx, bx, by, w, h, r);
      ctx.fill();

      ctx.strokeStyle = "#aaa";
      ctx.lineWidth = 1.5;
      roundRectPath(ctx, bx, by, w, h, r);
      ctx.stroke();

      ctx.fillStyle = classColors[cls];
      ctx.fillText(cls, bx + paddingX, y);

      legendItems.push({
        cls,
        x: bx,
        y: by,
        w,
        h
      });

      x += w + 12;
    });
  }

  function drawTimeTicks(w: number) {
    const DATE_LABEL_Y = 2;
    const TIME_LABEL_Y = 20;
    const TICK_TOP_Y = 36;
    const HEADER_H = 48;

    ctx.fillStyle = "#888";
    ctx.font = "12px sans-serif";

    const { start, end } = getTimelineBounds();
    const usableWidth = w - computedLeftMargin;
    const totalSeconds = end - start;

    const d = new Date(start * 1000);
    d.setHours(0, 0, 0, 0);
    let dayTs = d.getTime() / 1000;

    ctx.font = "13px sans-serif";
    ctx.fillStyle = "#ccc";
    ctx.textBaseline = "top";

    while (dayTs < end) {
      const x = xFor(dayTs);

      const label = d.toLocaleDateString(undefined, {
        weekday: "short",
        month: "short",
        day: "numeric"
      });

      ctx.fillText(label, x + 4, DATE_LABEL_Y);

      d.setDate(d.getDate() + 1);
      dayTs = d.getTime() / 1000;
    }

    const secondsPerPixel = totalSeconds / usableWidth;
    const TARGET_TICK_SPACING_PX = 50;
    const minTickInterval = secondsPerPixel * TARGET_TICK_SPACING_PX;

    const niceIntervals = [
      60,
      120,
      300,
      600,
      900,
      1800,
      3600,
      7200,
      14400
    ];

    let tickStep =
      niceIntervals.find((v) => v >= minTickInterval) ||
      niceIntervals[niceIntervals.length - 1];

    let labelEvery = tickStep;

    const labelFormat = (ts: number) => {
      const d = new Date(ts * 1000);
      return (
        d.getHours().toString().padStart(2, "0") +
        ":" +
        d.getMinutes().toString().padStart(2, "0")
      );
    };

    let t = Math.ceil(start / tickStep) * tickStep;

    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#888";

    const pxPerSecond = usableWidth / totalSeconds;

    const sampleLabel = labelFormat(start);
    const labelWidth = ctx.measureText(sampleLabel).width;

    let lastLabelRight = -Infinity;

    while (t < end) {
      const x = xFor(t);
      const isLabelTick = t % labelEvery === 0;

      if (isLabelTick) {
        const label = labelFormat(t);
        const wLabel = ctx.measureText(label).width;

        if (x > lastLabelRight + 8) {
          ctx.fillText(label, x + 4, TIME_LABEL_Y);
          lastLabelRight = x + 4 + wLabel;
        }
      }

      ctx.fillRect(
        x,
        TICK_TOP_Y,
        1,
        canvas.height - TICK_TOP_Y - LEGEND_HEIGHT
      );

      t += tickStep;
    }
  }

  function computeDynamicLeftMargin() {
    if (!ctx || !cameras || cameras.length === 0) return 50;

    ctx.font = "14px 'JetBrains Mono', monospace";

    let maxWidth = 0;
    for (const camera of cameras) {
      const w = ctx.measureText(camera.name).width;
      if (w > maxWidth) maxWidth = w;
    }

    return Math.ceil(maxWidth + 8);
  }

  function drawCameraRows(w: number) {
    ctx.font = "12px sans-serif";

    for (let i = 0; i < cameras.length; i++) {
      const y = HEADER_HEIGHT + i * ROW_HEIGHT;

      ctx.strokeStyle = "#333";
      ctx.strokeRect(computedLeftMargin, y, w - computedLeftMargin, ROW_HEIGHT);

      ctx.fillStyle = "#ccc";
      ctx.textBaseline = "middle";
      ctx.fillText(cameras[i].name, 2, y + ROW_HEIGHT / 2);
    }
  }

  function cameraRowIndex(name: string) {
    return cameras.findIndex((c) => c.name === name);
  }

  function drawEvents(w: number) {
    const usableWidth = w - computedLeftMargin;
    const timelineHeight = canvas.height - LEGEND_HEIGHT;
    const minWidth = 5;
    const eventBorderStyle = "#DDDDDD";
    const selectedEventBorderStyle = "#FFFFFF";

    ctx.save();
    ctx.beginPath();
    ctx.rect(
      computedLeftMargin,
      HEADER_HEIGHT,
      usableWidth,
      timelineHeight - HEADER_HEIGHT
    );
    ctx.clip();

    const { start, end } = getTimelineBounds();

    $eventStore.forEach((ev) => {
      const row = cameraRowIndex(ev.camera);
      if (row === -1) return;

      if (ev.end_time < start || ev.start_time > end) return;

      if (selectedClasses.size > 0) {
        const evClasses = Object.keys(ev.tags || {});
        const matches = evClasses.some((cls) => selectedClasses.has(cls));
        if (!matches) return;
      }

      const x1 = xFor(ev.start_time);
      const x2 = xFor(ev.end_time);
      const width = Math.max(2, x2 - x1);
      const y = HEADER_HEIGHT + row * ROW_HEIGHT;

      const tags = ev.tags || {};
      const evClasses = Object.keys(tags);

      if (evClasses.length === 0) {
        ctx.fillStyle = "#0f0";
        ctx.fillRect(x1, y + 5, width, ROW_HEIGHT - 10);

        if (width > minWidth) {
          ctx.lineWidth = 1;
          ctx.strokeStyle = eventBorderStyle;
          ctx.strokeRect(
            x1 + 0.5,
            y + 5 + 0.5,
            width - 1,
            ROW_HEIGHT - 10 - 1
          );
        }
        return;
      }

      const stripeHeight = (ROW_HEIGHT - 10) / evClasses.length;

      evClasses.forEach((cls, i) => {
        ctx.fillStyle = classColors[cls] || "#fff";
        ctx.fillRect(
          x1,
          y + 5 + i * stripeHeight,
          width,
          stripeHeight
        );
      });

      if (width > minWidth) {
        ctx.lineWidth = ev.start_time === selectedEventId ? 3 : 1;
        ctx.strokeStyle =
          ev.start_time === selectedEventId
            ? selectedEventBorderStyle
            : eventBorderStyle;
        ctx.strokeRect(
          x1 + 0.5,
          y + 5 + 0.5,
          width - 1,
          ROW_HEIGHT - 10 - 1
        );
      }
    });

    ctx.restore();
  }

  function handleHover(e: PointerEvent | MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;

    hoverEvent = findEventAt(mouseX, mouseY);
  }

  $: if (hoverEvent) updateTooltipPosition();

  async function updateTooltipPosition() {
    await tick();

    const margin = 12;

    const tooltipEl = document.querySelector(".tooltip");
    if (!tooltipEl) return;

    const tooltipRect = tooltipEl.getBoundingClientRect();
    const wrapperEl = document.querySelector(".timeline-wrapper");
    const wrapperRect = wrapperEl.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();

    const mouseXInWrapper = mouseX + (canvasRect.left - wrapperRect.left);
    const mouseYInWrapper = mouseY + (canvasRect.top - wrapperRect.top);

    let left = mouseXInWrapper + margin;
    let top = mouseYInWrapper + margin;

    if (left + tooltipRect.width > wrapperRect.width) {
      left = mouseXInWrapper - tooltipRect.width - margin;
    }

    if (left < margin) left = margin;

    if (top + tooltipRect.height > wrapperRect.height) {
      top = wrapperRect.height - tooltipRect.height - margin;
    }
    if (top < margin) top = margin;

    tooltipLeft = left;
    tooltipTop = top;
  }

  function findEventAt(x: number, y: number) {
    const { start, end } = getTimelineBounds();

    return $eventStore.find((ev) => {
      const row = cameraRowIndex(ev.camera);
      if (row === -1) return false;

      const rowY = HEADER_HEIGHT + row * ROW_HEIGHT;
      if (y < rowY || y > rowY + ROW_HEIGHT) return false;

      if (selectedClasses.size > 0) {
        const evClasses = Object.keys(ev.tags || {});
        const matches = evClasses.some((cls) => selectedClasses.has(cls));
        if (!matches) return false;
      }

      const x1 = xFor(ev.start_time);
      const x2 = xFor(ev.end_time);

      return x >= x1 && x <= x2;
    });
  }

  async function updateWindowAndFetch() {
    await fetchEventsForCurrentWindow();
    drawTimeline();
  }

  function onPointerDown(e: PointerEvent) {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (e.pointerType === "touch") {
      startX = e.clientX;
      startY = e.clientY;
      gestureMode = "pending";
    } else {
      canvas.setPointerCapture(e.pointerId);
    }

    if (pointers.size === 1) {
      panStartX = e.clientX;
      panStartOffset = offsetSeconds;
      isDragging = false;

      tapStart = { x: e.clientX, y: e.clientY, time: performance.now() };
      handleHover(e);
    }

    if (pointers.size === 2) {
      const pts = [...pointers.values()];
      zoomStartY = (pts[0].y + pts[1].y) / 2;
      zoomStartHours = zoomHours;

      const centerX = (pts[0].x + pts[1].x) / 2;
      zoomCenterX = centerX;

      const w = canvas.clientWidth;
      const usableWidth = w - computedLeftMargin;
      const pxPerSecond = usableWidth / (zoomHours * HOUR);

      zoomCenterSeconds =
        offsetSeconds + (centerX - computedLeftMargin) / pxPerSecond;
    }
  }

  function onPointerMove(e: PointerEvent) {
    if (e.pointerType === "mouse" && e.buttons === 0) {
      handleHover(e);
      return;
    }

    if (!pointers.has(e.pointerId)) return;

    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (e.pointerType === "touch" && pointers.size === 1) {
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;

      if (gestureMode === "pending") {
        if (Math.abs(dx) > 10 && Math.abs(dx) > Math.abs(dy)) {
          gestureMode = "pan-x";
          canvas.setPointerCapture(e.pointerId);
        } else if (Math.abs(dy) > 10) {
          gestureMode = "pan-y";
          return;
        } else {
          return;
        }
      }

      if (gestureMode === "pan-y") return;
    }

    if (pointers.size === 1) {
      const dx = e.clientX - panStartX;

      if (!isDragging) {
        if (Math.abs(dx) < 3) {
          handleHover(e);
          return;
        }
        isDragging = true;
      }

      const w = canvas.clientWidth;
      const usableWidth = w - computedLeftMargin;
      const pxPerSecond = usableWidth / (zoomHours * HOUR);

      offsetSeconds = panStartOffset + dx / pxPerSecond;
      offsetSeconds = Math.max(0, offsetSeconds);

      updateWindowAndFetch();
      return;
    }

    if (pointers.size === 2) {
      const pts = [...pointers.values()];
      const centerY = (pts[0].y + pts[1].y) / 2;
      const dy = centerY - zoomStartY;

      const zoomFactor = 1 - dy / 300;
      let newZoom = zoomStartHours * zoomFactor;

      newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));

      const w = canvas.clientWidth;
      const usableWidth = w - computedLeftMargin;
      const pxPerSecondBefore = usableWidth / (zoomHours * HOUR);

      const midSeconds =
        zoomCenterSeconds -
        (zoomCenterX - computedLeftMargin) / pxPerSecondBefore;

      zoomHours = newZoom;

      const pxPerSecondAfter = usableWidth / (zoomHours * HOUR);

      offsetSeconds =
        midSeconds +
        (zoomCenterX - computedLeftMargin) / pxPerSecondAfter;

      offsetSeconds = Math.max(0, offsetSeconds);

      updateWindowAndFetch();
    }
  }

  function onPointerUp(e: PointerEvent) {
    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }

    pointers.delete(e.pointerId);

    if (e.pointerType === "touch") {
      gestureMode = "none";
    }

    if (!isDragging && tapStart && pointers.size === 0) {
      const dx = Math.abs(e.clientX - tapStart.x);
      const dy = Math.abs(e.clientY - tapStart.y);
      const dt = performance.now() - tapStart.time;

      if (dx < 10 && dy < 10 && dt < 250) {
        handleTap(e.clientX, e.clientY);
      }
    }

    if (pointers.size === 0) {
      isDragging = false;
      tapStart = null;
    }
  }

  function onPointerCancel(e: PointerEvent) {
    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }

    pointers.clear();

    isDragging = false;
    gestureMode = "none";
    tapStart = null;

    if (e.pointerType === "mouse") {
      handleHover(e);
    }
  }

  function handleTap(clientX: number, clientY: number) {
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    if (y >= canvas.height - LEGEND_HEIGHT) {
      if (handleLegendClick(x, y)) return;
    }

    const ev = findEventAt(x, y);
    if (ev) {
      selectedEventId = ev.start_time;
      onSelectEvent(ev);
      drawTimeline();
    }
  }

  async function handleWheel(e: WheelEvent) {
    if (isMobile) return;
    if (!e.shiftKey) return;

    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;

    const w = canvas.clientWidth;
    const usableWidth = w - computedLeftMargin;

    const { start } = getTimelineBounds();
    const pxPerSecond = usableWidth / (zoomHours * HOUR);

    const timeAtCursor = start + (x - computedLeftMargin) / pxPerSecond;

    const zoomFactor = e.deltaY < 0 ? 0.95 : 1.05;
    let newZoom = zoomHours * zoomFactor;
    newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));

    const newPxPerSecond = usableWidth / (newZoom * HOUR);
    const newStart = timeAtCursor - (x - computedLeftMargin) / newPxPerSecond;

    const newOffset = serverNow - newStart - newZoom * HOUR;

    offsetSeconds = Math.max(0, newOffset);
    zoomHours = newZoom;

    await updateWindowAndFetch();
  }

  function hasVisibleEvents() {
    const { start, end } = getTimelineBounds();

    return $eventStore.some((ev) => {
      const row = cameraRowIndex(ev.camera);
      if (row === -1) return false;

      if (ev.end_time < start || ev.start_time > end) return false;

      if (selectedClasses.size > 0) {
        const evClasses = Object.keys(ev.tags || {});
        const matches = evClasses.some((cls) => selectedClasses.has(cls));
        if (!matches) return false;
      }

      return true;
    });
  }

  function drawNoEventsMessage() {
    ctx.save();
    ctx.fillStyle = "#ccc";
    ctx.font = "20px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("No Events", canvas.width / 2, canvas.height / 2);
    ctx.restore();
  }

  onDestroy(() => {
    if (serverTimeInterval) clearInterval(serverTimeInterval);
    if (ro) ro.disconnect();
  });

    //    on:pointerleave={onPointerCancel}
  //  on:pointerout={onPointerCancel}
</script>

<h3 class="timeline-title">Recorded Events</h3>
<div class="timeline-wrapper">
  <canvas
    bind:this={canvas}
    on:pointerdown={onPointerDown}
    on:pointermove={onPointerMove}
    on:pointerup={onPointerUp}
    on:pointercancel={onPointerUp}
    on:wheel={handleWheel}
    style="
      width: 100%;
      height: {HEADER_HEIGHT + cameras.length * ROW_HEIGHT + LEGEND_HEIGHT}px;
    "
  ></canvas>
  {#if hoverEvent}
    <div
      class="tooltip"
      style="left: {tooltipLeft}px; top: {tooltipTop}px;"
      >
      <div class="tooltip-title">{hoverEvent.camera}</div>
      <div class="tooltip-times">
        <div><strong>Start:</strong> {new Date(hoverEvent.start_time * 1000).toLocaleString()}</div>
        <div><strong>Stop:</strong> {new Date(hoverEvent.end_time * 1000).toLocaleString()}</div>
        <div><strong>Duration:</strong> {(hoverEvent.end_time - hoverEvent.start_time).toFixed(1)}s</div>
        <div><strong>Recorder:</strong> {hoverEvent.recorder_type}</div>
      </div>
      <div class="tooltip-classes">
        {#each Object.entries(hoverEvent.tags || {}) as [cls, detail]}
          <div class="tooltip-class-row">
            <span class="tooltip-class-name">{cls}</span>:
            <span class="tooltip-class-colors">{detail.join(", ")}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  :global(canvas) {
    touch-action: none !important;
  }

  canvas {
    border: 1px solid #666;
    background: #111;
    cursor: default;
    /*touch-action: pan-y !important;*/
  }

  .timeline-wrapper {
    position: relative;
    overflow: visible;
    z-index: 1;
  }

  .tooltip {
    position: absolute;
    background: #111;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    color: #eee;
    font-family: "Fira Code", "JetBrains Mono", Consolas, monospace;
    font-size: 0.85rem;
    pointer-events: none;
    z-index: 99999;
    max-width: 260px;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .tooltip-times div {
    margin: 0;
    padding: 0;
  }

  .tooltip-title {
    font-weight: bold;
    border-bottom: 1px solid #333;
    padding-bottom: 0.25rem;
    margin-bottom: 0.25rem;
  }

  .tooltip-class-row {
    display: flex;
    gap: 4px;
    white-space: normal;
  }

  .tooltip-class-name {
    font-weight: bold;
    color: #9cf;
  }

  .tooltip-class-colors {
    color: #eee;
  }

  .timeline-title {
    margin: 0;
    padding: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: bold;
    color: #eee;
    font-family: inherit;
  }
</style>
