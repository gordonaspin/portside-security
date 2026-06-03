<script>
  import { onMount, onDestroy, tick } from "svelte";
  import { debug, log, error } from "$lib/stores/logging";

  export let onSelectEvent = () => {};
  const TICK_HEIGHT = 20;
  const ROW_HEIGHT = 40;
  const HEADER_HEIGHT = TICK_HEIGHT + 36;  // extra band for date header
  const LEGEND_HEIGHT = 24;
  const HOUR = 3600;
  const DAY = 24 * HOUR;
  const isMobile =
    typeof window !== "undefined" &&
    /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  const MIN_ZOOM = 1/60
  const MAX_ZOOM = isMobile ? 4 : 24;

  let cameras = [];
  let events = [];
  let classes = [];
  let classColors = {};
  let selectedClasses = new Set();
  let canvas;
  let ctx;
  let initialAligned = false;
  let zoomHours = 24;          // desktop: 24h window; mobile: overridden to 4h on mount
  let offsetSeconds = 0;       // desktop: aligned to latest event; mobile: live edge (0)
  let computedLeftMargin = 140;
  let serverNow = 0;
  let legendItems = [];
  let selectedEventId = null;

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
    // newest event is at the END (oldest → newest)
    const newestStart = events.length > 0
      ? events[events.length - 1].start_time
      : null;

    const url = newestStart
      ? `/api/events?mobile=${isMobile ? 1 : 0}&since=${newestStart}`
      : `/api/events?mobile=${isMobile ? 1 : 0}`;

    const res = await fetch(url, { credentials: "include" });
    const data = await res.json();

    // map new events
    const newEvents = data.events.map((rec) => {
      const media_url_encoded = rec.media_filename
        .split("/")
        .map(encodeURIComponent)
        .join("/");
      const metadata_url_encoded = rec.metadata_filename
        .split("/")
        .map(encodeURIComponent)
        .join("/");

      return {
        ...rec,
        media_url: "/" + media_url_encoded,
        metadata_url: "/" + metadata_url_encoded
      };
    });

    // append because list is oldest → newest
    events = [...events, ...newEvents];

    // Desktop: align right edge to latest event on initial load only
    if (!initialAligned) {
      if (!isMobile && events.length > 0) {
        const latestEnd = Math.max(...events.map((e) => e.end_time));
        offsetSeconds = Math.max(0, serverNow - latestEnd);
      }

      if (isMobile) {
        zoomHours = 4;
        offsetSeconds = 0; // live edge
      }

      initialAligned = true;
    }
  }


  function getTimelineBounds() {
    const end = serverNow - offsetSeconds;
    const start = end - zoomHours * HOUR;
    return { start, end };
  }

  function xFor(ts) {
    const { start, end } = getTimelineBounds();
    const total = end - start;
    return computedLeftMargin + ((ts - start) / total) * (canvas.width - computedLeftMargin);
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
    if (!hasVisibleEvents()) {
      drawNoEventsMessage();
    }
  }

  function drawBackground(w, h) {
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, w, h);
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

      // Background (only difference between selected/unselected)
      ctx.fillStyle = isSelected
        ? "rgba(255,255,255,0.36)"   // selected
        : "rgba(255,255,255,0.06)";  // unselected
      roundRectPath(ctx, bx, by, w, h, r);
      ctx.fill();

      // Single consistent border
      ctx.strokeStyle = "#aaa";   // light neutral border
      ctx.lineWidth = 1.5;
      roundRectPath(ctx, bx, by, w, h, r);
      ctx.stroke();

      // Text
      ctx.fillStyle = classColors[cls];
      ctx.fillText(cls, bx + paddingX, y);

      // Hit-test region
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

  function roundRectPath(ctx, x, y, w, h, r) {
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


  function handleLegendClick(mx, my) {
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
        drawTimeline();
        return true;
      }
    }
    return false;
  }

  function drawTimeTicks(w) {
    
    const DATE_LABEL_Y = 2;       // day/date
    const TIME_LABEL_Y = 20;      // HH:MM
    const TICK_TOP_Y   = 36;      // tick lines start here
    const HEADER_HEIGHT = 48; // enough room for both label bands
    
    ctx.fillStyle = "#888";
    ctx.font = "12px sans-serif";

    const { start, end } = getTimelineBounds();
    const usableWidth = w - computedLeftMargin;
    const totalSeconds = end - start;

    //
    // 1. Draw day/date headers at local midnights
    //
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

    //
    // Determine tickStep dynamically based on pixel spacing
    //
    const secondsPerPixel = totalSeconds / usableWidth;
    const TARGET_TICK_SPACING_PX = 50;
    const minTickInterval = secondsPerPixel * TARGET_TICK_SPACING_PX;

    // “Nice” intervals in seconds
    const niceIntervals = [
      60,        // 1 min
      120,       // 2 min
      300,       // 5 min
      600,       // 10 min
      900,       // 15 min
      1800,      // 30 min
      3600,      // 1 hour
      7200,      // 2 hours
      14400      // 4 hours
    ];

    // Pick the smallest nice interval >= minTickInterval
    let tickStep = niceIntervals.find(v => v >= minTickInterval) || niceIntervals[niceIntervals.length - 1];

    // Labels every tickStep (or every other if needed)
    let labelEvery = tickStep;

    const labelFormat = (ts) => {
      const d = new Date(ts * 1000);
      return (
        d.getHours().toString().padStart(2, "0") +
        ":" +
        d.getMinutes().toString().padStart(2, "0")
      );
    };

    //
    // 3. Draw hour/minute ticks (label-width aware, overlap-free)
    //
    let t = Math.ceil(start / tickStep) * tickStep;
    let k = 0;

    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#888";

    const pxPerSecond = usableWidth / totalSeconds;
    const pxPerTick = tickStep * pxPerSecond;

    // Measure a sample label
    const sampleLabel = labelFormat(start);
    const labelWidth = ctx.measureText(sampleLabel).width;

    // Track last drawn label position
    let lastLabelRight = -Infinity;

    while (t < end) {
      const x = xFor(t);

      const isLabelTick = (t % labelEvery === 0);

      //
      // LABEL DRAWING (no overlap)
      //
      if (isLabelTick) {
        const label = labelFormat(t);
        const w = ctx.measureText(label).width;

        // Only draw if it won't overlap previous label
        if (x > lastLabelRight + 8) {
          ctx.fillText(label, x + 4, TIME_LABEL_Y);
          lastLabelRight = x + 4 + w;
        }
      }

      //
      // TICK DRAWING (always allowed)
      //
      ctx.fillRect(
        x,
        TICK_TOP_Y,
        1,
        canvas.height - TICK_TOP_Y - LEGEND_HEIGHT
      );

      t += tickStep;
      k += 1;
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

  function drawCameraRows(w) {
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

  function cameraRowIndex(name) {
    return cameras.findIndex((c) => c.name === name);
  }

  function drawEvents(w) {
    const usableWidth = w - computedLeftMargin;
    const timelineHeight = canvas.height - LEGEND_HEIGHT;
    const minWidth = 5;
    const eventBorderStyle = "#DDDDDD";
    const selectedEventBorderStyle = "#FFFFFF";

    ctx.save();
    ctx.beginPath();
    ctx.rect(computedLeftMargin, HEADER_HEIGHT, usableWidth, timelineHeight - HEADER_HEIGHT);
    ctx.clip();

    const { start, end } = getTimelineBounds();

    events.forEach((ev) => {
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
          ctx.strokeRect(x1 + 0.5, y + 5 + 0.5, width - 1, (ROW_HEIGHT - 10) - 1);
        }
        return;
      }

      const stripeHeight = (ROW_HEIGHT - 10) / evClasses.length;

      evClasses.forEach((cls, i) => {
        ctx.fillStyle = classColors[cls] || "#fff";
        ctx.fillRect(x1, y + 5 + i * stripeHeight, width, stripeHeight);
      });

      if (width > minWidth) {
        ctx.lineWidth = ev.start_time === selectedEventId ? 3 : 1;
        ctx.strokeStyle = ev.start_time === selectedEventId ? selectedEventBorderStyle : eventBorderStyle;
        ctx.strokeRect(x1 + 0.5, y + 5 + 0.5, width - 1, (ROW_HEIGHT - 10) - 1);
      }
    });

    ctx.restore();
  }

  // Hover
  let hoverEvent = null;
  let mouseX = 0;
  let mouseY = 0;
  let tooltipLeft = 0;
  let tooltipTop = 0;

  function handleHover(e) {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;

    hoverEvent = findEventAt(mouseX, mouseY);
  }

  $: if (hoverEvent) updateTooltipPosition();

  async function updateTooltipPosition() {
      await tick(); // wait for tooltip to render

      const margin = 12;

      // Tooltip element
      const tooltipEl = document.querySelector(".tooltip");
      if (!tooltipEl) return;

      const tooltipRect = tooltipEl.getBoundingClientRect();

      // Timeline wrapper element (tooltip is positioned relative to this)
      const wrapperEl = document.querySelector(".timeline-wrapper");
      const wrapperRect = wrapperEl.getBoundingClientRect();

      // Convert mouse coords (canvas-relative) to viewport coords
      const canvasRect = canvas.getBoundingClientRect();
      const mouseViewportX = canvasRect.left + mouseX;
      const mouseViewportY = canvasRect.top + mouseY;

      // Default position (to the right of cursor)
      let left = mouseViewportX + margin;
      let top = mouseViewportY + margin;

      // Clamp horizontally (flip if needed)
      if (left + tooltipRect.width > window.innerWidth) {
          left = mouseViewportX - tooltipRect.width - margin;
      }
      if (left < margin) {
          left = margin;
      }

      // Clamp vertically
      if (top + tooltipRect.height > window.innerHeight) {
          top = window.innerHeight - tooltipRect.height - margin;
      }
      if (top < margin) {
          top = margin;
      }

      // Convert viewport coords → wrapper coords
      tooltipLeft = left - wrapperRect.left;
      tooltipTop = top - wrapperRect.top;
  }

  function findEventAt(x, y) {
    const { start, end } = getTimelineBounds();

    return events.find((ev) => {
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

  // Pointer / gestures
  let gestureMode = "none"; // "pending" | "pan-x" | "pan-y"
  let startX = 0;
  let startY = 0;

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
    // Track pointer
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    // --- MOBILE / TOUCH LOGIC ---
    if (e.pointerType === "touch") {
      // Start gesture direction detection
      startX = e.clientX;
      startY = e.clientY;
      gestureMode = "pending";   // "pending" → "pan-x" or "pan-y"
      // IMPORTANT: do NOT capture yet
    }

    // --- DESKTOP / MOUSE LOGIC ---
    else {
      // Mouse can safely capture immediately
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

  function onPointerMove(e) {
    // --- MOUSE HOVER PATH ---
    if (e.pointerType === "mouse" && e.buttons === 0) {
      handleHover(e);
      return;
    }

    if (!pointers.has(e.pointerId)) return;

    // Update pointer position
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    // --- TOUCH / MOBILE LOGIC (direction detection) ---
    if (e.pointerType === "touch" && pointers.size === 1) {
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;

      // Determine gesture direction
      if (gestureMode === "pending") {
        if (Math.abs(dx) > 10 && Math.abs(dx) > Math.abs(dy)) {
          gestureMode = "pan-x";
          canvas.setPointerCapture(e.pointerId);   // capture ONLY now
        } else if (Math.abs(dy) > 10) {
          gestureMode = "pan-y";   // allow browser scroll
          return;
        } else {
          return; // not enough movement yet
        }
      }

      // Vertical scroll → let browser handle it
      if (gestureMode === "pan-y") {
        return;
      }
      // Horizontal pan → fall through to pan logic
    }

    // --- ONE-FINGER PAN (mouse or touch pan-x) ---
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

      requestAnimationFrame(drawTimeline);
      return;
    }

    // --- TWO-FINGER ZOOM ---
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
        zoomCenterSeconds - (zoomCenterX - computedLeftMargin) / pxPerSecondBefore;

      zoomHours = newZoom;

      const pxPerSecondAfter = usableWidth / (zoomHours * HOUR);

      offsetSeconds =
        midSeconds + (zoomCenterX - computedLeftMargin) / pxPerSecondAfter;

      offsetSeconds = Math.max(0, offsetSeconds);

      requestAnimationFrame(drawTimeline);
    }
  }


  function onPointerUp(e) {
    // Release capture if we had it
    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }

    // Remove pointer from active set
    pointers.delete(e.pointerId);

    // --- reset gesture mode for touch ---
    if (e.pointerType === "touch") {
        gestureMode = "none";
    }

    // Tap detection
    if (!isDragging && tapStart && pointers.size === 0) {
      const dx = Math.abs(e.clientX - tapStart.x);
      const dy = Math.abs(e.clientY - tapStart.y);
      const dt = performance.now() - tapStart.time;

      if (dx < 10 && dy < 10 && dt < 250) {
        handleTap(e.clientX, e.clientY);
      }
    }

    // Reset drag state
    if (pointers.size === 0) {
      isDragging = false;
      tapStart = null;
    }
  }

  function handleTap(clientX, clientY) {
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
      drawTimeline()
    }
  }

  function handleWheel(e) {
    if (isMobile) {
        return; // ignore scroll
    }

    if (!e.shiftKey) {
        return; // ignore scroll
    }

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

    requestAnimationFrame(drawTimeline);
  }

  // Lifecycle
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
      zoomHours = 4;   // 4h window
      offsetSeconds = 0; // live edge at now
    }
    zoomHours = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoomHours));

    computedLeftMargin = computeDynamicLeftMargin();
    drawTimeline();

    ro = new ResizeObserver(() => {
      computedLeftMargin = computeDynamicLeftMargin();
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
    }, 5000);
  });

  function hasVisibleEvents() {
    const { start, end } = getTimelineBounds();

    return events.some((ev) => {
      // camera exists?
      const row = cameraRowIndex(ev.camera);
      if (row === -1) return false;

      // time overlap with window?
      if (ev.end_time < start || ev.start_time > end) return false;

      // class filter?
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
    if (loadEventsInterval) clearInterval(loadEventsInterval);
    if (ro) ro.disconnect();
  });
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
