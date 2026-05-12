<script>
  import { onMount, onDestroy } from 'svelte';
  const log = window.mosaic.log;
  export let onSelectEvent = () => {};

  let cameras = [];
  let events = [];
  let classes = [];
  let classColors = {};

  let canvas;
  let ctx;

  // SERVER TIME
  let serverNow = 0;
  const HOUR = 3600;
  const DAY = 24 * HOUR;

  const isMobile = typeof window !== "undefined" &&
    /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

  let zoomHours = 24;

  function clampZoom() {
    const maxHours = isMobile ? 4 : 24;
    const minHours = isMobile ? 0.05 : 0.25; // ⭐ 3 minutes on mobile
    zoomHours = Math.max(minHours, Math.min(maxHours, zoomHours));
  }

  clampZoom();

  let offsetSeconds = 0;   // now in seconds, not hours

  let isTouchPanning = false;
  let isPinching = false;

  let lastTouchX = 0;

  let pinchStartCenterY = 0;
  let pinchStartZoom = 0;
  let pinchCenterX = 0;
  let pinchCenterSeconds = 0;
  let tapStartX = 0;
  let tapStartY = 0;
  let tapStartTime = 0;
  let touchMoved = false;

  let LEFT_MARGIN = 140;
  const MIN_ZOOM_HOURS = 0.25;
  const MAX_ZOOM_HOURS = 24;
  const TICK_HEIGHT = 20;
  const ROW_HEIGHT = 40;
  const HEADER_HEIGHT = TICK_HEIGHT + 20;
  const LEGEND_HEIGHT = 24;


  // ----------------------------------------
  // Load server time
  // ----------------------------------------
  async function loadServerTime() {
    const res = await fetch('/api/server_time', { credentials: "include" });
    const data = await res.json();
    serverNow = data.epoch;
  }

  // ----------------------------------------
  // Load backend data
  // ----------------------------------------
  async function loadClasses() {
    const res = await fetch('/api/classes', { credentials: "include" });
    const data = await res.json();
    classes = data.classes;

    const palette = [
      "#ff4444", "#4488ff", "#aa44ff", "#ffff44",
      "#ff8844", "#aa6633", "#44ff44", "#44ffff"
    ];

    classColors = {};
    classes.forEach((cls, i) => {
      classColors[cls] = palette[i % palette.length];
    });
  }

  async function loadCameras() {
    const res = await fetch('/api/cameras', { credentials: "include" });
    cameras = await res.json();
  }

  async function loadEvents() {
    const res = await fetch(`/api/events?mobile=${isMobile ? 1 : 0}`, { credentials: "include" });
    const data = await res.json();

    events = data.events.map(rec => {
      const video_url_encoded = rec.output.split("/").map(encodeURIComponent).join("/");
      const metadata_url_encoded = rec.metadata.split("/").map(encodeURIComponent).join("/");

      return {
        ...rec,
        video_url: "/" + video_url_encoded,
        metadata_url: "/" + metadata_url_encoded
      };
    });
  }

  // ----------------------------------------
  // Timeline bounds (server time)
  // ----------------------------------------
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

  // ----------------------------------------
  // Drawing
  // ----------------------------------------
  function drawTimeline() {
    if (!ctx) return;

    const w = canvas.width = canvas.clientWidth;
    const h = canvas.height = canvas.clientHeight;

    ctx.clearRect(0, 0, w, h);

    // Determine tick spacing based on zoom level and device width
    let majorTick = 3600;   // 1 hour
    let minorTick = 900;    // 15 min
    let labelEvery = 3600;  // 1 hour

    const isMobile = window.innerWidth < 700;

    // Zoom-based thinning
    if (zoomHours > 8) {
      majorTick = 7200;     // 2 hours
      minorTick = 0;        // hide minor ticks
      labelEvery = 7200;
    }

    if (zoomHours > 12) {
      majorTick = 14400;    // 4 hours
      minorTick = 0;
      labelEvery = 14400;
    }

    // Mobile-specific thinning
    if (isMobile) {
      if (zoomHours > 4) {
        majorTick = 7200;   // 2 hours
        minorTick = 0;
        labelEvery = 7200;
      }
      if (zoomHours > 8) {
        majorTick = 14400;  // 4 hours
        minorTick = 0;
        labelEvery = 14400;
      }
    }

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

  // ----------------------------------------
  // Legend
  // ----------------------------------------
  function drawLegend() {
    ctx.font = "12px sans-serif";
    ctx.textBaseline = "top";

    const y = canvas.height - LEGEND_HEIGHT + 4;
    let x = LEFT_MARGIN;

    classes.forEach(cls => {
      ctx.fillStyle = classColors[cls];
      ctx.fillText(cls, x, y);
      x += ctx.measureText(cls).width + 20;
    });
  }

  // ----------------------------------------
  // Time ticks (server time)
  // ----------------------------------------
  function drawTimeTicks(w) {
    ctx.fillStyle = "#888";
    ctx.font = "12px sans-serif";

    const { start, end } = getTimelineBounds();
    const usableWidth = w - LEFT_MARGIN;
    const totalSeconds = end - start;

    const isMobile = window.innerWidth < 700;

    // --- Determine tick spacing based on zoom + mobile ---
    let tickStep;      // seconds between ticks
    let labelEvery;    // seconds between labels
    let labelFormat;

    if (zoomHours >= 12) {
      // Very zoomed out
      tickStep = 4 * 3600;      // 4 hours
      labelEvery = 4 * 3600;
    } else if (zoomHours >= 6) {
      tickStep = 2 * 3600;      // 2 hours
      labelEvery = 2 * 3600;
    } else if (zoomHours >= 4) {
      tickStep = 3600;          // 1 hour
      labelEvery = 3600;
    } else if (zoomHours >= 1) {
      tickStep = 900;           // 15 minutes
      labelEvery = isMobile ? 1800 : 900; // mobile: fewer labels
    } else {
      tickStep = 60;            // 1 minute
      labelEvery = isMobile ? 300 : 120; // mobile: fewer labels
    }

    // --- Label formatter ---
    labelFormat = ts => {
      const d = new Date(ts * 1000);
      return (
        d.getHours().toString().padStart(2, "0") +
        ":" +
        d.getMinutes().toString().padStart(2, "0")
      );
    };

    // --- Find first tick >= start ---
    let t = Math.ceil(start / tickStep) * tickStep;

    // --- Draw ticks ---
    while (t < end) {
      const x = xFor(t);

      // Skip ticks that would overlap on mobile
      if (isMobile) {
        const pxPerTick = (tickStep / totalSeconds) * usableWidth;
        if (pxPerTick < 40) {
          // too dense → skip every other tick
          if ((t / tickStep) % 2 !== 0) {
            t += tickStep;
            continue;
          }
        }
      }

      // Draw label only at labelEvery interval
      if (t % labelEvery === 0) {
        ctx.fillText(labelFormat(t), x + 4, 2);
      }

      // Tick line
      ctx.fillRect(
        x,
        TICK_HEIGHT,
        1,
        canvas.height - TICK_HEIGHT - LEGEND_HEIGHT
      );

      t += tickStep;
    }
  }

  // ----------------------------------------
  // Camera rows
  // ----------------------------------------
  function computeDynamicLeftMargin() {
    if (!ctx || !cameras || cameras.length === 0) return 120; // fallback

    ctx.font = "14px 'JetBrains Mono', monospace";

    let maxWidth = 0;
    for (const cam of cameras) {
      const w = ctx.measureText(cam.name).width;
      if (w > maxWidth) maxWidth = w;
    }

    return Math.ceil(maxWidth + 20); // +20px padding
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
    return cameras.findIndex(c => c.name === name);
  }

  // ----------------------------------------
  // Events (multi-class stripes)
  // ----------------------------------------
  function drawEvents(w) {
    const usableWidth = w - LEFT_MARGIN;
    const timelineHeight = canvas.height - LEGEND_HEIGHT;

    ctx.save();
    ctx.beginPath();
    ctx.rect(LEFT_MARGIN, HEADER_HEIGHT, usableWidth, timelineHeight - HEADER_HEIGHT);
    ctx.clip();

    const { start, end } = getTimelineBounds();

    events.forEach(ev => {
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
  // Hover tooltip
  // ----------------------------------------
  let hoverEvent = null;
  let mouseX = 0;
  let mouseY = 0;

  function findEventAt(x, y) {
    const { start, end } = getTimelineBounds();

    return events.find(ev => {
      const row = cameraRowIndex(ev.camera);
      if (row === -1) return false;

      const rowY = HEADER_HEIGHT + row * ROW_HEIGHT;
      if (y < rowY || y > rowY + ROW_HEIGHT) return false;

      const x1 = xFor(ev.start_time);
      const x2 = xFor(ev.end_time);

      return x >= x1 && x <= x2;
    });
  }

  function getDistance(t1, t2) {
    const dx = t2.clientX - t1.clientX;
    const dy = t2.clientY - t1.clientY;
    return Math.sqrt(dx*dx + dy*dy);
  }

  function getCenter(t1, t2) {
    return {
      x: (t1.clientX + t2.clientX) / 2,
      y: (t1.clientY + t2.clientY) / 2
    };
  }

  function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;

    hoverEvent = findEventAt(mouseX, mouseY);
  }

  // ----------------------------------------
  // Click
  // ----------------------------------------
  function handleClick(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const ev = findEventAt(x, y);
    if (ev) onSelectEvent(ev);
  }

  function handleTap(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    const ts = timeForX(x);          // uses LEFT_MARGIN internally
    const camIndex = Math.floor((y - HEADER_HEIGHT) / ROW_HEIGHT);
    if (camIndex < 0) return;

    const ev = findEventAt(camIndex, ts);
    if (ev) {
      onSelectEvent(ev); // same as desktop
    }
  }

  // ----------------------------------------
  // Panning (converted to seconds)
  // ----------------------------------------
  let isPanning = false;
  let panStartX = 0;
  let panStartOffset = 0;

  function handlePanStart(e) {
    isPanning = true;
    panStartX = e.clientX;
    panStartOffset = offsetSeconds;
  }

  function handleMouseMoveCombined(e) {
    handleMouseMove(e);
    handlePanMove(e);
  }

  function handlePanMove(e) {
    if (!isPanning) return;

    const w = canvas.clientWidth;
    const usableWidth = w - LEFT_MARGIN;
    const pxPerSecond = usableWidth / (zoomHours * HOUR);

    const dx = e.clientX - panStartX;
    const deltaSeconds = dx / pxPerSecond;

    offsetSeconds = Math.max(0, Math.min(DAY - zoomHours * HOUR, panStartOffset + deltaSeconds));
    drawTimeline();
  }

  function handlePanEnd() {
    isPanning = false;
  }

  // ----------------------------------------
  // Zoom
  // ----------------------------------------
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
    newZoom = Math.max(MIN_ZOOM_HOURS, Math.min(MAX_ZOOM_HOURS, newZoom));

    const newPxPerSecond = usableWidth / (newZoom * HOUR);
    const newStart = timeAtCursor - (x - LEFT_MARGIN) / newPxPerSecond;

    const newOffset = (serverNow - newStart) - newZoom * HOUR;

    offsetSeconds = Math.max(0, Math.min(DAY - newZoom * HOUR, newOffset));
    zoomHours = newZoom;
    clampZoom();

    drawTimeline();
  }

  function onTouchStart(e) {
    if (e.touches.length === 1) {
      const t = e.touches[0];

      // One-finger pan
      isTouchPanning = true;
      isPinching = false;
      lastTouchX = t.clientX;

      // Tap detection start
      tapStartX = t.clientX;
      tapStartY = t.clientY;
      tapStartTime = performance.now();
      touchMoved = false;

      return;
    }

    else if (e.touches.length === 2) {
      // Two-finger vertical zoom
      isTouchPanning = false;
      isPinching = true;

      // ⭐ Reset tap state so it doesn't leak into touchend
      touchMoved = true;

      const [t1, t2] = e.touches;

      pinchStartCenterY = (t1.clientY + t2.clientY) / 2;
      pinchStartZoom = zoomHours;

      const c = getCenter(t1, t2);
      pinchCenterX = c.x;

      // Convert pinch center to timeline seconds
      const w = canvas.clientWidth;
      const usableWidth = w - LEFT_MARGIN;
      const pxPerSecond = usableWidth / (zoomHours * HOUR);
      pinchCenterSeconds = offsetSeconds + (c.x - LEFT_MARGIN) / pxPerSecond;

      return;
    }
  }


  function onTouchMove(e) {
    // ⭐ ONE-FINGER PAN
    if (isTouchPanning && e.touches.length === 1) {
      const t = e.touches[0];

      const dx = t.clientX - lastTouchX;
      lastTouchX = t.clientX;

      // Mark as moved so tap doesn't fire
      if (Math.abs(dx) > 3) {
        touchMoved = true;
      }

      // Convert dx to seconds
      const w = canvas.clientWidth;
      const usableWidth = w - LEFT_MARGIN;
      const pxPerSecond = usableWidth / (zoomHours * HOUR);

      offsetSeconds += dx / pxPerSecond;

      if (isMobile) {
        const minOffset = Math.max(0, DAY - 4 * HOUR);
        const maxOffset = DAY - zoomHours * HOUR;
        offsetSeconds = Math.max(minOffset, Math.min(maxOffset, offsetSeconds));
      } else {
        offsetSeconds = Math.max(0, Math.min(DAY - zoomHours * HOUR, offsetSeconds));
      }

      requestAnimationFrame(drawTimeline);
      e.preventDefault();
      return;
    }

    // ⭐ TWO-FINGER VERTICAL ZOOM (no pinch zoom)
    if (isPinching && e.touches.length === 2) {
      const [t1, t2] = e.touches;

      // Gesture center Y
      const centerY = (t1.clientY + t2.clientY) / 2;

      // Vertical drag distance
      const dy = centerY - pinchStartCenterY;

      // Convert vertical drag to zoom
      const zoomFactor = 1 - dy / 300; // tune sensitivity
      zoomHours = pinchStartZoom * zoomFactor;

      // Clamp zoom
      const minZoomHours = isMobile ? 0.05 : 0.25;
      zoomHours = Math.max(minZoomHours, Math.min(24, zoomHours));
      clampZoom();

      // Maintain center-based zooming
      const w = canvas.clientWidth;
      const usableWidth = w - LEFT_MARGIN;
      const pxPerSecond = usableWidth / (zoomHours * HOUR);

      offsetSeconds =
        pinchCenterSeconds - (pinchCenterX - LEFT_MARGIN) / pxPerSecond;

      if (isMobile) {
        const minOffset = Math.max(0, DAY - 4 * HOUR);
        const maxOffset = DAY - zoomHours * HOUR;
        offsetSeconds = Math.max(minOffset, Math.min(maxOffset, offsetSeconds));
      } else {
        offsetSeconds = Math.max(0, Math.min(DAY - zoomHours * HOUR, offsetSeconds));
      }

      requestAnimationFrame(drawTimeline);
      e.preventDefault();
      return;
    }
  }

  function onTouchEnd(e) {
    if (e.changedTouches.length === 1) {
      const t = e.changedTouches[0];

      const dx = Math.abs(t.clientX - tapStartX);
      const dy = Math.abs(t.clientY - tapStartY);
      const dt = performance.now() - tapStartTime;

      const isTap = !touchMoved && dx < 10 && dy < 10 && dt < 250;

      if (isTap) {
        handleTap(t.clientX, t.clientY);
        return;
      }
    }

    isTouchPanning = false;
    isPinching = false;
  }

  // ----------------------------------------
  // Lifecycle
  // ----------------------------------------
  let ro;
  let interval;

  onMount(async () => {
    ctx = canvas.getContext("2d");

    await loadServerTime();
    await loadClasses();
    await loadCameras();
    await loadEvents();
    LEFT_MARGIN = computeDynamicLeftMargin();
    drawTimeline();

    ro = new ResizeObserver(() => {
      LEFT_MARGIN = computeDynamicLeftMargin();
      drawTimeline()
    });
    ro.observe(canvas);

    // Refresh server time every minute
    setInterval(async () => {
      await loadServerTime();
      drawTimeline();
    }, 60000);

    interval = setInterval(async () => {
      await loadEvents();
      drawTimeline();
    }, 15000);
  });

  onDestroy(() => {
    if (interval) clearInterval(interval);
    if (ro) ro.disconnect();
  });
</script>

<div class="timeline-wrapper">
  <canvas
    id="timeline"
    bind:this={canvas}
    on:click={handleClick}
    on:mousemove={handleMouseMove}
    on:mousedown={handlePanStart}
    on:mousemove={handlePanMove}
    on:mouseup={handlePanEnd}
    on:mouseleave={handlePanEnd}
    on:wheel={handleWheel}
    on:touchstart={onTouchStart}
    on:touchmove={onTouchMove}
    on:touchend={onTouchEnd}
    on:touchcancel={onTouchEnd}
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
