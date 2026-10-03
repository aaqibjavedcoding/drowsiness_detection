/* ------------------------------------------------------------------
 * Drowsiness Detection — browser client
 *
 * The browser owns the camera (getUserMedia) and the UI; the Flask server
 * owns the detection pipeline (MediaPipe Face Mesh + EAR/MAR/head pose +
 * DrowsinessEngine). Frames go up as JPEG, state comes back as JSON.
 * ------------------------------------------------------------------ */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    video: $("video"),
    overlay: $("overlay"),
    stage: $("stage"),
    placeholder: $("placeholder"),
    hud: $("hud"),
    hudEar: $("hud-ear"),
    hudMar: $("hud-mar"),
    hudHead: $("hud-head"),
    hudScore: $("hud-score"),
    noface: $("noface"),
    flash: $("flash"),
    start: $("btn-start"),
    stop: $("btn-stop"),
    reset: $("btn-reset"),
    alarm: $("btn-alarm"),
    mesh: $("toggle-mesh"),
    status: $("status"),
    pillFps: $("pill-fps"),
    pillEngine: $("pill-engine"),
    banner: $("state-banner"),
    stateText: $("state-text"),
    stateSub: $("state-sub"),
    gauge: $("gauge-value"),
    scoreNumber: $("score-number"),
    rEar: $("r-ear"),
    rEye: $("r-eye"),
    rClosure: $("r-closure"),
    rMar: $("r-mar"),
    rMouth: $("r-mouth"),
    rHead: $("r-head"),
    sBlink: $("s-blink"),
    sLong: $("s-long"),
    sYawn: $("s-yawn"),
    sWarn: $("s-warn"),
    sDrowsy: $("s-drowsy"),
    sMaxClose: $("s-maxclose"),
    sDuration: $("s-duration"),
    toast: $("toast"),
    toastTitle: $("toast-title"),
    toastText: $("toast-text"),
    toastLink: $("toast-link"),
    toastClose: $("toast-close"),
  };

  const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52;
  const SEND_INTERVAL_MS = 110; // ~9 fps round-trips; plenty for EAR/MAR
  const CAPTURE_WIDTH = 480;

  const state = {
    sessionId: null,
    stream: null,
    running: false,
    inFlight: false,
    timer: null,
    alarmEnabled: true,
    mesh: null,
    lastRoundTrip: 0,
    fps: 0,
    lastTick: 0,
  };

  const capture = document.createElement("canvas");
  const captureCtx = capture.getContext("2d", { willReadFrequently: true });
  const ctx = els.overlay.getContext("2d");

  /* ---------------------------------------------------------------- *
   * Alarm (WebAudio — no asset download, works offline)
   * ---------------------------------------------------------------- */
  const alarm = {
    audioCtx: null,
    osc: null,
    gain: null,
    playing: false,
    unlock() {
      if (!this.audioCtx) {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) return;
        this.audioCtx = new Ctx();
      }
      if (this.audioCtx.state === "suspended") this.audioCtx.resume();
    },
    start() {
      if (this.playing || !state.alarmEnabled) return;
      this.unlock();
      if (!this.audioCtx) return;
      const t = this.audioCtx.currentTime;
      this.osc = this.audioCtx.createOscillator();
      this.gain = this.audioCtx.createGain();
      const lfo = this.audioCtx.createOscillator();
      const lfoGain = this.audioCtx.createGain();

      this.osc.type = "square";
      this.osc.frequency.value = 760;
      lfo.frequency.value = 4.5;        // pulsing beep-beep-beep
      lfoGain.gain.value = 0.16;
      this.gain.gain.value = 0.16;

      lfo.connect(lfoGain).connect(this.gain.gain);
      this.osc.connect(this.gain).connect(this.audioCtx.destination);
      this.osc.start(t);
      lfo.start(t);
      this._lfo = lfo;
      this.playing = true;
    },
    stop() {
      if (!this.playing) return;
      try { this.osc.stop(); this._lfo.stop(); } catch (_) {}
      try { this.osc.disconnect(); this.gain.disconnect(); } catch (_) {}
      this.playing = false;
    },
  };

  /* ---------------------------------------------------------------- *
   * Helpers
   * ---------------------------------------------------------------- */
  function setStatus(text, isError = false) {
    els.status.textContent = text;
    els.status.classList.toggle("error", isError);
  }

  function showToast(title, text, withLink = false) {
    els.toastTitle.textContent = title;
    els.toastText.textContent = text;
    els.toastLink.href = window.location.href;
    els.toastLink.classList.toggle("hidden", !withLink);
    els.toast.classList.remove("hidden");
  }

  function setEnginePill(live, label) {
    els.pillEngine.innerHTML =
      `<i class="dot ${live ? "dot-live" : "dot-idle"}"></i> ${label}`;
  }

  function fmtDuration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }

  function tagClass(el, cls, text) {
    el.className = "tag " + cls;
    el.textContent = text;
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw Object.assign(new Error(data.error || res.statusText), { data, status: res.status });
    return data;
  }

  /* ---------------------------------------------------------------- *
   * Camera
   * ---------------------------------------------------------------- */
  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("insecure-context");
    }
    return navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    els.video.srcObject = null;
  }

  function cameraErrorMessage(err) {
    const name = err && err.name;
    if (err && err.message === "insecure-context") {
      return [
        "Camera API not available",
        "The browser only exposes the webcam over HTTPS or on localhost. Open this page " +
          "on https:// or at http://localhost:5000 and try again.",
        true,
      ];
    }
    if (name === "NotAllowedError" || name === "SecurityError") {
      return [
        "Camera permission blocked",
        "Your browser denied camera access. Allow the camera for this page (padlock icon → " +
          "Camera → Allow), then press Start again. If this page is inside an embedded preview " +
          "frame, open it in a new tab — embedded frames often block the camera.",
        true,
      ];
    }
    if (name === "NotFoundError" || name === "OverconstrainedError") {
      return ["No camera found", "No webcam was detected on this device. Connect one and press Start again.", false];
    }
    if (name === "NotReadableError") {
      return ["Camera is busy", "Another application (Zoom, Meet, Teams, another tab…) is using the webcam. Close it and press Start again.", false];
    }
    return ["Camera error", (err && err.message) || String(err), false];
  }

  /* ---------------------------------------------------------------- *
   * Overlay drawing
   * ---------------------------------------------------------------- */
  async function loadMesh() {
    if (state.mesh) return state.mesh;
    try {
      const res = await fetch("/api/mesh");
      state.mesh = await res.json();
    } catch (_) {
      state.mesh = { contours: [], left_eye: [], right_eye: [], lips: [] };
    }
    return state.mesh;
  }

  function resizeOverlay() {
    const rect = els.stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    els.overlay.width = Math.round(rect.width * dpr);
    els.overlay.height = Math.round(rect.height * dpr);
  }

  function drawOverlay(points, result) {
    const w = els.overlay.width;
    const h = els.overlay.height;
    ctx.clearRect(0, 0, w, h);
    if (!points || !points.length || !els.mesh.checked) return;

    // The video uses object-fit: cover — replicate that mapping so the mesh
    // sits exactly on the face.
    const vw = els.video.videoWidth || 4;
    const vh = els.video.videoHeight || 3;
    const scale = Math.max(w / vw, h / vh);
    const dx = (w - vw * scale) / 2;
    const dy = (h - vh * scale) / 2;
    const P = (i) => {
      const p = points[i];
      return p ? [p[0] * vw * scale + dx, p[1] * vh * scale + dy] : null;
    };

    const strokeEdges = (edges, color, width) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (const [a, b] of edges) {
        const pa = P(a), pb = P(b);
        if (!pa || !pb) continue;
        ctx.moveTo(pa[0], pa[1]);
        ctx.lineTo(pb[0], pb[1]);
      }
      ctx.stroke();
    };

    const mesh = state.mesh || {};
    strokeEdges(mesh.contours || [], "rgba(120, 190, 255, 0.45)", 1.2);

    const eyeColor = result.eye.closed ? "rgba(231,76,60,0.95)" : "rgba(46,204,113,0.95)";
    strokeEdges(mesh.left_eye || [], eyeColor, 2.4);
    strokeEdges(mesh.right_eye || [], eyeColor, 2.4);

    const mouthColor = result.mouth.yawning
      ? "rgba(243,156,18,0.95)"
      : "rgba(160,170,200,0.75)";
    strokeEdges(mesh.lips || [], mouthColor, 2.0);
  }

  /* ---------------------------------------------------------------- *
   * Frame loop
   * ---------------------------------------------------------------- */
  function grabFrameDataUrl() {
    const vw = els.video.videoWidth;
    const vh = els.video.videoHeight;
    if (!vw || !vh) return null;
    const scale = CAPTURE_WIDTH / vw;
    capture.width = CAPTURE_WIDTH;
    capture.height = Math.round(vh * scale);
    // Mirror the capture so the on-screen mirrored preview and the detected
    // landmarks share the same coordinate space.
    captureCtx.save();
    captureCtx.translate(capture.width, 0);
    captureCtx.scale(-1, 1);
    captureCtx.drawImage(els.video, 0, 0, capture.width, capture.height);
    captureCtx.restore();
    return capture.toDataURL("image/jpeg", 0.6);
  }

  async function tick() {
    if (!state.running || state.inFlight) return;
    const image = grabFrameDataUrl();
    if (!image) return;

    state.inFlight = true;
    try {
      const result = await postJSON("/api/frame", { session_id: state.sessionId, image });
      applyResult(result);

      const now = performance.now();
      if (state.lastTick) {
        const inst = 1000 / (now - state.lastTick);
        state.fps = state.fps ? state.fps * 0.8 + inst * 0.2 : inst;
        els.pillFps.textContent = `FPS ${state.fps.toFixed(1)}`;
      }
      state.lastTick = now;
    } catch (err) {
      if (err.status === 404) {
        // Server restarted or session expired — transparently re-create it.
        try {
          const s = await postJSON("/api/session/start");
          state.sessionId = s.session_id;
        } catch (_) { /* retried on the next tick */ }
      } else {
        setStatus(`Server error (recovered): ${err.message}`, true);
      }
    } finally {
      state.inFlight = false;
    }
  }

  function applyResult(r) {
    const state_ = r.state;
    const isDrowsy = state_ === "DROWSY";
    const isWarning = state_ === "WARNING";

    els.banner.classList.toggle("warning", isWarning);
    els.banner.classList.toggle("drowsy", isDrowsy);
    els.stateText.textContent = state_;
    els.stateSub.textContent = isDrowsy
      ? "Wake up! Take a break now"
      : isWarning
      ? "Early fatigue signs detected"
      : "Driver looks alert";

    const score = r.score;
    els.scoreNumber.textContent = Math.round(score);
    els.gauge.style.strokeDashoffset = String(
      GAUGE_CIRCUMFERENCE * (1 - Math.min(score, 100) / 100)
    );
    els.gauge.style.stroke = isDrowsy ? "#e74c3c" : isWarning ? "#f39c12" : "#2ecc71";

    els.flash.classList.toggle("on", isDrowsy);
    els.noface.classList.toggle("hidden", r.face_found);

    if (r.face_found) {
      els.rEar.textContent = r.eye.ear.toFixed(3);
      els.rMar.textContent = r.mouth.mar.toFixed(3);
      els.rClosure.textContent = `${r.eye.closure_duration.toFixed(2)} s`;
      tagClass(els.rEye, r.eye.long_closure ? "bad" : r.eye.closed ? "warn" : "ok",
        r.eye.long_closure ? "MICROSLEEP" : r.eye.closed ? "CLOSED" : "OPEN");
      tagClass(els.rMouth, r.mouth.yawning ? "bad" : r.mouth.open ? "warn" : "ok",
        r.mouth.yawning ? "YAWNING" : r.mouth.open ? "OPEN" : "CLOSED");
      tagClass(els.rHead, r.head.down || r.head.away ? "warn" : "ok",
        `${r.head.label} (p${r.head.pitch.toFixed(0)}/y${r.head.yaw.toFixed(0)})`);

      els.hudEar.textContent = r.eye.ear.toFixed(2);
      els.hudMar.textContent = r.mouth.mar.toFixed(2);
      els.hudHead.textContent = r.head.label;
      els.hudScore.textContent = Math.round(score);

      if (!r.no_face_timeout) setStatus("Monitoring in progress…");
    } else {
      [els.rEar, els.rMar, els.rClosure].forEach((el) => (el.textContent = "--"));
      [els.rEye, els.rMouth, els.rHead].forEach((el) => tagClass(el, "", "--"));
      els.hudEar.textContent = els.hudMar.textContent = "--";
      els.hudHead.textContent = "--";
      if (r.no_face_timeout) setStatus("No face detected — please face the camera.", true);
    }

    const s = r.stats;
    els.sBlink.textContent = Math.round(s.blink_count);
    els.sLong.textContent = Math.round(s.long_blink_count);
    els.sYawn.textContent = Math.round(s.yawn_count);
    els.sWarn.textContent = Math.round(s.warning_count);
    els.sDrowsy.textContent = Math.round(s.drowsy_count);
    els.sMaxClose.textContent = s.max_closure_duration.toFixed(2);
    els.sDuration.textContent = fmtDuration(s.session_duration);

    setEnginePill(true, state_);
    drawOverlay(r.landmarks, r);

    if (isDrowsy && state.alarmEnabled) alarm.start();
    else alarm.stop();
  }

  /* ---------------------------------------------------------------- *
   * Controls
   * ---------------------------------------------------------------- */
  async function onStart() {
    if (state.running) return;
    els.start.disabled = true;
    setStatus("Requesting camera access…");

    let stream;
    try {
      stream = await startCamera();
    } catch (err) {
      const [title, text, link] = cameraErrorMessage(err);
      showToast(title, text, link);
      setStatus(`${title}. ${text}`, true);
      els.start.disabled = false;
      return;
    }

    state.stream = stream;
    els.video.srcObject = stream;
    try { await els.video.play(); } catch (_) {}

    try {
      const s = await postJSON("/api/session/start");
      state.sessionId = s.session_id;
    } catch (err) {
      stopCamera();
      showToast("Server unavailable", "Could not reach the detection server. Is web_app.py still running?");
      setStatus("Could not reach the detection server.", true);
      els.start.disabled = false;
      return;
    }

    await loadMesh();
    resizeOverlay();
    alarm.unlock(); // user gesture — lets the alarm play later

    state.running = true;
    state.lastTick = 0;
    state.fps = 0;
    els.placeholder.classList.add("hidden");
    els.hud.classList.add("show");
    els.stop.disabled = false;
    els.start.disabled = true;
    setStatus("Monitoring in progress…");
    setEnginePill(true, "Live");

    state.timer = setInterval(tick, SEND_INTERVAL_MS);
  }

  async function onStop(message = "Stopped — statistics are kept. Press Start to resume.") {
    if (!state.running) return;
    state.running = false;
    clearInterval(state.timer);
    state.timer = null;
    stopCamera();
    alarm.stop();
    ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);

    els.placeholder.classList.remove("hidden");
    els.hud.classList.remove("show");
    els.flash.classList.remove("on");
    els.noface.classList.add("hidden");
    els.start.disabled = false;
    els.stop.disabled = true;
    els.pillFps.textContent = "FPS --";
    setEnginePill(false, "Paused");
    setStatus(message);
  }

  async function onReset() {
    if (!state.sessionId) {
      [els.sBlink, els.sLong, els.sYawn, els.sWarn, els.sDrowsy].forEach((el) => (el.textContent = "0"));
      els.sMaxClose.textContent = "0.00";
      els.sDuration.textContent = "00:00";
      setStatus("Session statistics have been reset.");
      return;
    }
    try {
      await postJSON("/api/session/reset", { session_id: state.sessionId });
      [els.sBlink, els.sLong, els.sYawn, els.sWarn, els.sDrowsy].forEach((el) => (el.textContent = "0"));
      els.sMaxClose.textContent = "0.00";
      els.sDuration.textContent = "00:00";
      els.scoreNumber.textContent = "0";
      els.gauge.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
      els.gauge.style.stroke = "#2ecc71";
      els.banner.classList.remove("warning", "drowsy");
      els.stateText.textContent = "NORMAL";
      els.stateSub.textContent = "Driver looks alert";
      els.flash.classList.remove("on");
      alarm.stop();
      setStatus("Session statistics have been reset.");
    } catch (err) {
      setStatus(`Could not reset the session: ${err.message}`, true);
    }
  }

  function onToggleAlarm() {
    state.alarmEnabled = !state.alarmEnabled;
    els.alarm.setAttribute("aria-pressed", String(state.alarmEnabled));
    els.alarm.innerHTML = state.alarmEnabled
      ? '<span class="btn-ico">🔊</span> Alarm: On'
      : '<span class="btn-ico">🔇</span> Alarm: Off';
    if (!state.alarmEnabled) alarm.stop();
    else alarm.unlock();
  }

  /* ---------------------------------------------------------------- *
   * Wiring
   * ---------------------------------------------------------------- */
  els.start.addEventListener("click", onStart);
  els.stop.addEventListener("click", () => onStop());
  els.reset.addEventListener("click", onReset);
  els.alarm.addEventListener("click", onToggleAlarm);
  els.toastClose.addEventListener("click", () => els.toast.classList.add("hidden"));
  els.mesh.addEventListener("change", () => {
    if (!els.mesh.checked) ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  });
  window.addEventListener("resize", resizeOverlay);
  window.addEventListener("beforeunload", () => {
    if (state.sessionId && navigator.sendBeacon) {
      navigator.sendBeacon(
        "/api/session/stop",
        new Blob([JSON.stringify({ session_id: state.sessionId })], { type: "application/json" })
      );
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === " ") { e.preventDefault(); state.running ? onStop() : onStart(); }
    if (e.key.toLowerCase() === "r") onReset();
  });

  resizeOverlay();
  setEnginePill(false, "Idle");

  // Friendly heads-up when the page can't ever get a camera (http:// on a
  // remote host) — tell the user before they press a button that can't work.
  if (!window.isSecureContext) {
    setStatus(
      "Heads-up: browsers only grant camera access on https:// or localhost. " +
        "Open this page over https or on http://localhost:5000.",
      true
    );
  }
})();
