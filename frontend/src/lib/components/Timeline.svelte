<script>
  import { onMount, onDestroy } from "svelte";

  export let onSelectEvent = () => {};

  let cameras = [];
  let events = [];
  let classes = [];
  let classColors = {};

  let canvas;
  let ctx;

  const HOUR = 3600;
  const DAY = 24 * HOUR;

  const isMobile =
    typeof window !== "undefined" &&
    /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

  let zoomHours = 24;
  let offsetSeconds = 0;

  const MIN_ZOOM = isMobile ? 0.05 : 0.25;
  const MAX_ZOOM = isMobile ? 4 : 24;

  let LEFT_MARGIN = 140;
  const TICK_HEIGHT = 20;
  const ROW_HEIGHT = 40;
  const HEADER_HEIGHT = TICK_HEIGHT + 20;
  const LEGEND_HEIGHT = 24;

  let serverNow = 0;

  async function loadServerTime() {
    const res = await fetch("/api/server_time", { credentials: "include" });
    const data = await res.json();
    serverNow = data.epoch;
  }

  async function loadClasses() {
    const res = await fetch("/api/classes", { credentials: "include" });
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
    const res = await fetch("/api/cameras", { credentials: "include" });
    cameras = await res.json();
  }

  async function loadEvents() {
    const res = await fetch(
      `/api/events?mobile=${isMobile ? 1 : 0}`,
      { credentials: "include" }
    );
    const data = await res.json();

    events = data.events.map((rec) => {
      const video_url_encoded = rec.output
        .split("/")
        .map(encodeURIComponent)
        .join("/");
      const metadata_url_encoded = rec.metadata
        .split("/")
        .map(encodeURIComponent)
        .join("/");

      return {
        ...rec,
        video_url: "/" + video_url_encoded,
        metadata_url: "/" + metadata_url_encoded
      };
    });
  }

  function getTimelineBounds() {
    const end = serverNow - offsetSeconds;
    const start = end - zoomHours * HOUR;
    return { start, end };
  }

  function xFor(ts) {
    const { start, end } = getTimelineBounds();
    const total = end - start;
    return LEFT_MARGIN + ((ts - start) / total) * (canvas.width - LEFT_MARGIN);
  }

  function drawTimeline() {
    if (!ctx) return;

    const w = (canvas.width = canvas.clientWidth);
    const h = (canvas.height = canvas.clientHeight);

    ctx.clearRect(0, 0, w, h);

    drawBackground(w, h);
    drawLegend();
    drawTimeTicks(w);
    drawCameraRows(w);
    drawEvents(w);
  }

  function drawBackground(w, h) {
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, w, h);
  }

  function drawLegend() {
    ctx.font = "12px sans-serif";
    ctx.textBaseline = "top";

    const y = canvas.height - LEGEND_HEIGHT + 4;
    let x = LEFT_MARGIN;

    classes.forEach((cls) => {
      ctx.fillStyle = classColors[cls];
      ctx.fillText(cls, x, y);
      x += ctx.measureText(cls).width + 20;
    });
  }

  function drawTimeTicks(w) {
    ctx.fillStyle = "#888";
    ctx.font = "12px sans-serif";

    const { start, end } = getTimelineBounds();
    const usableWidth = w - LEFT_MARGIN;
    const totalSeconds = end - start;

    const isMobileView = window.innerWidth < 700;

    let tickStep;
    let labelEvery;

    if (zoomHours >= 12) {
      tickStep = 4 * 3600;
      labelEvery = 4 * 3600;
    } else if (zoomHours >= 6) {
      tickStep = 2 * 3600;
      labelEvery = 2 * 3600;
    } else if (zoomHours >= 4) {
      tickStep = 3600;
      labelEvery = 3600;
    } else if (zoomHours >= 1) {
      tickStep = 900;
      labelEvery = isMobileView ? 1800 : 900;
    } else {
      tickStep = 60;
      labelEvery = isMobileView ? 300 : 120;
    }

    const labelFormat = (ts) => {
      const d = new Date(ts * 1000);
      return (
        d.getHours().toString().padStart(2, "0") +
        ":" +
        d.getMinutes().toString().padStart(2, "0")
      );
    };

    let t = Math.ceil(start / tickStep) * tickStep;

    while (t < end) {
      const x = xFor(t);

      if (isMobileView) {
        const pxPerTick = (tickStep / totalSeconds) * usableWidth;
        if (pxPerTick < 40) {
          if ((t / tickStep) % 2 !== 0) {
            t += tickStep;
            continue;
          }
        }
      }

      if (t % labelEvery === 0) {
        ctx.fillText(labelFormat(t), x + 4, 2);
      }

      ctx.fillRect(
        x,
        TICK_HEIGHT,
        1,
        canvas.height - TICK_HEIGHT - LEGEND_HEIGHT
      );

      t += tickStep;
    }
  }

  function computeDynamicLeftMargin() {
    if (!ctx || !cameras || cameras.length === 0) return 120;

    ctx.font = "14px 'JetBrains Mono', monospace";

    let maxWidth = 0;
    for (const cam of cameras) {
      const w = ctx.measureText(cam.name).width;
      if (w > maxWidth) maxWidth = w;
    }

    return Math.ceil(maxWidth + 20);
  }

  function drawCameraRows(w) {
    ctx.font = "12px sans-serif";

    for (let i = 0; i < cameras.length; i++) {
      const y = HEADER_HEIGHT + i * ROW_HEIGHT;

      ctx.strokeStyle = "#333";
      ctx.strokeRect(LEFT_MARGIN, y, w - LEFT_MARGIN, ROW_HEIGHT);

      ctx.fillStyle = "#ccc";
      ctx.textBaseline = "middle";
      ctx.fillText(cameras[i].name, 10, y + ROW_HEIGHT / 2);
    }
  }

  function cameraRowIndex(name) {
    return cameras.findIndex((c) => c.name === name);
  }

  function drawEvents(w) {
    const usableWidth = w - LEFT_MARGIN;
    const timelineHeight = canvas.height - LEGEND_HEIGHT;

    ctx.save();
    ctx.beginPath();
    ctx.rect(LEFT_MARGIN, HEADER_HEIGHT, usableWidth, timelineHeight - HEADER_HEIGHT);
    ctx.clip();

    const { start, end } = getTimelineBounds();

    events.forEach((ev) => {
      const row = cameraRowIndex(ev.camera);
      if (row === -1) return;

      if (ev.end_time < start || ev.start_time > end) return;

      const x1 = xFor(ev.start_time);
      const x2 = xFor(ev.end_time);
      const width = Math.max(2, x2 - x1);
      const y = HEADER_HEIGHT + row * ROW_HEIGHT;

      const tags = ev.tags || {};
      const evClasses = Object.keys(tags);

      if (evClasses.length === 0) {
        ctx.fillStyle = "#0f0";
        ctx.fillRect(x1, y + 5, width, ROW_HEIGHT - 10);
        return;
      }

      const stripeHeight = (ROW_HEIGHT - 10) / evClasses.length;

      evClasses.forEach((cls, i) => {
        ctx.fillStyle = classColors[cls] || "#fff";
        ctx.fillRect(x1, y + 5 + i * stripeHeight, width, stripeHeight);
      });
    });

    ctx.restore();
  }

  // ----------------------------------------
  // Hover
  // ----------------------------------------
  let hoverEvent = null;
  let mouseX = 0;
  let mouseY = 0;

  function handleHover(e) {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;

    hoverEvent = findEventAt(mouseX, mouseY);
  }

  function findEventAt(x, y) {
    const { start, end } = getTimelineBounds();

    return events.find((ev) => {
      const row = cameraRowIndex(ev.camera);
      if (row === -1) return false;

      const rowY = HEADER_HEIGHT + row * ROW_HEIGHT;
      if (y < rowY || y > rowY + ROW_HEIGHT) return false;

      const x1 = xFor(ev.start_time);
      const x2 = xFor(ev.end_time);

      return x >= x1 && x <= x2;
    });
  }

  // ----------------------------------------
  // Pointer Events — Unified Gestures
  // ----------------------------------------
  let pointers = new Map();
  let panStartX = 0;
  let panStartOffset = 0;
  let isDragging = false;

  let zoomStartY = 0;
  let zoomStartHours = 0;
  let zoomCenterX = 0;
  let zoomCenterSeconds = 0;

  let tapStart = null;

  function onPointerDown(e) {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    canvas.setPointerCapture(e.pointerId);

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
      const usableWidth = w - LEFT_MARGIN;
      const pxPerSecond = usableWidth / (zoomHours * HOUR);

      zoomCenterSeconds =
        offsetSeconds + (centerX - LEFT_MARGIN) / pxPerSecond;
    }
  }

  function onPointerMove(e) {
    // Pure hover (mouse move, no buttons)
    if (e.pointerType === "mouse" && e.buttons === 0) {
      handleHover(e);
      return;
    }

    if (!pointers.has(e.pointerId)) return;

    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    // Single pointer: hover or pan
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
      const usableWidth = w - LEFT_MARGIN;
      const pxPerSecond = usableWidth / (zoomHours * HOUR);

      offsetSeconds = panStartOffset + dx / pxPerSecond;

      const minOffset = isMobile ? DAY - 4 * HOUR : 0;
      const maxOffset = DAY - zoomHours * HOUR;
      offsetSeconds = Math.max(minOffset, Math.min(maxOffset, offsetSeconds));

      requestAnimationFrame(drawTimeline);
      return;
    }

    // Two pointers: vertical zoom
    if (pointers.size === 2) {
      const pts = [...pointers.values()];
      const centerY = (pts[0].y + pts[1].y) / 2;
      const dy = centerY - zoomStartY;

      const zoomFactor = 1 - dy / 300;
      let newZoom = zoomStartHours * zoomFactor;

      newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));

      const w = canvas.clientWidth;
      const usableWidth = w - LEFT_MARGIN;
      const pxPerSecondBefore = usableWidth / (zoomHours * HOUR);

      const midSeconds =
        zoomCenterSeconds - (zoomCenterX - LEFT_MARGIN) / pxPerSecondBefore;

      zoomHours = newZoom;

      const pxPerSecondAfter = usableWidth / (zoomHours * HOUR);

      offsetSeconds =
        midSeconds + (zoomCenterX - LEFT_MARGIN) / pxPerSecondAfter;

      const minOffset = isMobile ? DAY - 4 * HOUR : 0;
      const maxOffset = DAY - zoomHours * HOUR;
      offsetSeconds = Math.max(minOffset, Math.min(maxOffset, offsetSeconds));

      requestAnimationFrame(drawTimeline);
    }
  }

  function onPointerUp(e) {
    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
    pointers.delete(e.pointerId);

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

  function handleTap(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    const ev = findEventAt(x, y);
    if (ev) onSelectEvent(ev);
  }

  function handleWheel(e) {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;

    const w = canvas.clientWidth;
    const usableWidth = w - LEFT_MARGIN;

    const { start } = getTimelineBounds();
    const pxPerSecond = usableWidth / (zoomHours * HOUR);

    const timeAtCursor = start + (x - LEFT_MARGIN) / pxPerSecond;

    const zoomFactor = e.deltaY < 0 ? 0.95 : 1.05;
    let newZoom = zoomHours * zoomFactor;
    newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));

    const newPxPerSecond = usableWidth / (newZoom * HOUR);
    const newStart = timeAtCursor - (x - LEFT_MARGIN) / newPxPerSecond;

    const newOffset = serverNow - newStart - newZoom * HOUR;

    offsetSeconds = Math.max(0, Math.min(DAY - newZoom * HOUR, newOffset));
    zoomHours = newZoom;

    requestAnimationFrame(drawTimeline);
  }

  // ----------------------------------------
  // Lifecycle
  // ----------------------------------------
  let serverTimeInterval;
  let loadEventsInterval;
  let ro;

  onMount(async () => {
    ctx = canvas.getContext("2d");

    await loadServerTime();
    await loadClasses();
    await loadCameras();
    await loadEvents();

    if (isMobile) {
      zoomHours = 4;
      offsetSeconds = DAY - 4 * HOUR;
    }
    
    LEFT_MARGIN = computeDynamicLeftMargin();
    drawTimeline();

    ro = new ResizeObserver(() => {
      LEFT_MARGIN = computeDynamicLeftMargin();
      drawTimeline();
    });
    ro.observe(canvas);

    serverTimeInterval = setInterval(async () => {
      await loadServerTime();
      drawTimeline();
    }, 60000);

    loadEventsInterval = setInterval(async () => {
      await loadEvents();
      drawTimeline();
    }, 15000);
  });

  onDestroy(() => {
    if (serverTimeInterval) clearInterval(serverTimeInterval);
    if (loadEventsInterval) clearInterval(loadEventsInterval);
    if (ro) ro.disconnect();
  });
</script>

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
      style="left: {mouseX + 12}px; top: {mouseY + 12}px;"
    >
      <div class="tooltip-title">{hoverEvent.camera}</div>

      <div class="tooltip-times">
        <div><strong>Start:</strong> {new Date(hoverEvent.start_time * 1000).toLocaleString()}</div>
        <div><strong>Stop:</strong> {new Date(hoverEvent.end_time * 1000).toLocaleString()}</div>
        <div><strong>Duration:</strong> {(hoverEvent.end_time - hoverEvent.start_time).toFixed(1)}s</div>
      </div>

      <div class="tooltip-classes">
        {#each Object.entries(hoverEvent.tags || {}) as [cls, colors]}
          <div class="tooltip-class-row">
            <span class="tooltip-class-name">{cls}</span>:
            <span class="tooltip-class-colors">{colors.join(", ")}</span>
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
    touch-action: none !important;
  }

  .timeline-wrapper {
    position: relative;
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
    white-space: nowrap;
  }

  .tooltip-class-name {
    font-weight: bold;
    color: #9cf;
  }

  .tooltip-class-colors {
    color: #eee;
  }
</style>
