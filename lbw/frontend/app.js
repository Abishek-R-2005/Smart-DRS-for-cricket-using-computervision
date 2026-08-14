/**
 * LBW Decision Review System — Frontend Logic
 */

(() => {
  "use strict";

  // ============ DOM References ============
  const $ = (sel) => document.querySelector(sel);

  // Hub Page
  const hubPage = $("#hub-page");
  const cardsLbw = document.querySelectorAll(".card-lbw");

  // Landing Page
  const landingPage = $("#landing-page");
  const landingLoader = $("#landing-loader");
  const btnEnter = $("#btn-enter");
  const landingAudio = $("#landing-audio");

  // Panel
  const uploadZone = $("#upload-zone");
  const videoInput = $("#video-input");
  const fileNameDisplay = $("#file-name-display");
  const btnUpload = $("#btn-upload");
  const uploadLoader = $("#upload-loader");
  const uploadBtnText = $("#upload-btn-text");
  const progressContainer = $("#progress-container");
  const progressBar = $("#progress-bar");

  const sectionUpload = $("#section-upload");
  const sectionFrames = $("#section-frames");
  const sectionTrajectory = $("#section-trajectory");
  const sectionDecision = $("#section-decision");

  const pitchFrameInput = $("#pitch-frame");
  const impactFrameInput = $("#impact-frame");
  const btnProcessFrames = $("#btn-process-frames");
  const framesLoader = $("#frames-loader");
  const framesBtnText = $("#frames-btn-text");

  const btnCompute = $("#btn-compute");
  const computeLoader = $("#compute-loader");
  const computeBtnText = $("#compute-btn-text");

  const pitchCoordsEl = $("#pitch-coords");
  const impactCoordsEl = $("#impact-coords");
  const dotPitch = $("#dot-pitch");
  const dotImpact = $("#dot-impact");
  const frameInstruction = $("#frame-instruction");

  // Popup
  const decisionPopup = $("#decision-popup");
  const popupBadge = $("#popup-badge");
  const popupTitle = $("#popup-title");
  const popupDetails = $("#popup-details");
  const btnClosePopup = $("#btn-close-popup");

  // DRS Sidebar
  const drsSidebar = $("#drs-sidebar");
  const drsItemDecision = $("#drs-status-decision");
  const drsItemWickets = $("#drs-status-wickets");
  const drsItemImpact = $("#drs-status-impact");
  const drsItemPitching = $("#drs-status-pitching");

  const valDecision = $("#val-decision");
  const valWickets = $("#val-wickets");
  const valImpact = $("#val-impact");
  const valPitching = $("#val-pitching");

  // Viewport
  const viewportEmpty = $("#viewport-empty");
  const stageVideo = $("#stage-video");
  const processedVideo = $("#processed-video");
  const stageFrames = $("#stage-frames");
  const stageTrajectory = $("#stage-trajectory");
  const trajectoryImg = $("#trajectory-img");

  const canvasPitch = $("#canvas-pitch");
  const canvasImpact = $("#canvas-impact");
  const crosshairPitch = $("#crosshair-pitch");
  const crosshairImpact = $("#crosshair-impact");
  const frameCardPitch = $("#frame-card-pitch");
  const frameCardImpact = $("#frame-card-impact");

  // Umpire video overlays
  const umpireOverlay = $("#umpire-overlay");
  const umpireVideo = $("#umpire-video");
  const outroOverlay = $("#outro-overlay");
  const outroVideo = $("#outro-video");

  // ============ State ============
  let sessionId = null;
  let frameCount = 0;
  let pitchClick = null;   // {x, y} in original image coords
  let impactClick = null;
  let pitchImgNatural = null;   // {w, h} natural size
  let impactImgNatural = null;
  let impactFrameNumber = 0;
  let stumpX1 = 0, stumpX2 = 0;

  // ============ Helpers ============
  function showStage(stage) {
    [viewportEmpty, stageVideo, stageFrames, stageTrajectory].forEach((s) =>
      s.classList.add("viewport__stage--hidden")
    );
    if (stage === "empty") viewportEmpty.classList.remove("viewport__stage--hidden");
    if (stage === "video") stageVideo.classList.remove("viewport__stage--hidden");
    if (stage === "frames") stageFrames.classList.remove("viewport__stage--hidden");
    if (stage === "trajectory") stageTrajectory.classList.remove("viewport__stage--hidden");
    // viewport-empty is not viewport__stage, handle separately
    if (stage === "empty") viewportEmpty.style.display = "";
    else viewportEmpty.style.display = "none";
  }

  function showSection(sec) {
    sec.classList.remove("panel__section--hidden");
  }

  function setLoading(btn, loader, textEl, isLoading, loadingText, normalText) {
    if (isLoading) {
      btn.classList.add("btn--loading");
      btn.disabled = true;
      textEl.textContent = loadingText;
    } else {
      btn.classList.remove("btn--loading");
      btn.disabled = false;
      textEl.textContent = normalText;
    }
  }

  function animateProgress(start, end, durationMs) {
    progressContainer.style.display = "";
    const startTime = performance.now();
    function step(now) {
      const elapsed = now - startTime;
      const t = Math.min(elapsed / durationMs, 1);
      const pct = start + (end - start) * t;
      progressBar.style.width = pct + "%";
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ============ Fullscreen Video Overlay Helper ============
  /**
   * Play a video in a fullscreen overlay, wait for it to finish, then fade out.
   * @param {HTMLElement} overlay  - the .umpire-overlay container
   * @param {HTMLVideoElement} video - the <video> inside it
   * @param {string} src - path to the video file
   * @returns {Promise<void>} resolves after fade-out is complete
   */
  function playOverlayVideo(overlay, video, src) {
    return new Promise((resolve) => {
      video.src = src;
      video.currentTime = 0;

      // Show overlay
      overlay.classList.remove("umpire-overlay--fading");
      overlay.classList.add("umpire-overlay--visible");

      video.play().catch(() => { /* autoplay may fail silently */ });

      // When video ends, fade out then resolve
      video.onended = () => {
        overlay.classList.add("umpire-overlay--fading");
        // Wait for CSS transition (0.8s)
        setTimeout(() => {
          overlay.classList.remove("umpire-overlay--visible", "umpire-overlay--fading");
          video.src = "";
          resolve();
        }, 900);
      };

      // Safety: if video errors out, just resolve so the flow isn't blocked
      video.onerror = () => {
        overlay.classList.remove("umpire-overlay--visible", "umpire-overlay--fading");
        video.src = "";
        resolve();
      };
    });
  }

  // ============ Drag & Drop ============
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
      videoInput.files = e.dataTransfer.files;
      handleFileSelect();
    }
  });

  // ============ File Select ============
  videoInput.addEventListener("change", handleFileSelect);

  function handleFileSelect() {
    const file = videoInput.files[0];
    if (!file) return;
    fileNameDisplay.textContent = file.name;
    btnUpload.disabled = false;
  }

  // ============ Upload ============
  btnUpload.addEventListener("click", async () => {
    const file = videoInput.files[0];
    if (!file) return;

    // --- Play umpire review intro video (rev1.mp4) first ---
    btnUpload.disabled = true;
    await playOverlayVideo(umpireOverlay, umpireVideo, "vid/rev1.mp4");

    setLoading(btnUpload, uploadLoader, uploadBtnText, true, "Processing…", "Upload & Process");
    animateProgress(0, 85, 8000);

    const formData = new FormData();
    formData.append("video", file);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      sessionId = data.session_id;
      frameCount = data.frame_count;

      // Set progress to 100
      progressBar.style.width = "100%";
      setTimeout(() => { progressContainer.style.display = "none"; }, 800);

      // Show video
      processedVideo.src = data.video_url;
      showStage("video");

      // Show frame selection
      pitchFrameInput.max = frameCount - 1;
      impactFrameInput.max = frameCount - 1;
      showSection(sectionFrames);
      btnProcessFrames.disabled = false;

      setLoading(btnUpload, uploadLoader, uploadBtnText, false, "", "Upload & Process");
      btnUpload.disabled = true; // already uploaded
      uploadBtnText.textContent = "✓ Uploaded";
    } catch (err) {
      console.error(err);
      alert("Upload failed: " + err.message);
      setLoading(btnUpload, uploadLoader, uploadBtnText, false, "", "Upload & Process");
      progressContainer.style.display = "none";
    }
  });

  // ============ Process Frames ============
  btnProcessFrames.addEventListener("click", async () => {
    const pf = parseInt(pitchFrameInput.value, 10);
    const imf = parseInt(impactFrameInput.value, 10);
    impactFrameNumber = imf;

    if (isNaN(pf) || isNaN(imf) || pf < 0 || imf < 0 || pf >= frameCount || imf >= frameCount) {
      alert("Please enter valid frame numbers (0 – " + (frameCount - 1) + ")");
      return;
    }

    setLoading(btnProcessFrames, framesLoader, framesBtnText, true, "Detecting…", "Extract & Detect Stumps");

    try {
      const [pitchRes, impactRes] = await Promise.all([
        fetch(`/api/frame/${sessionId}/${pf}`).then((r) => r.json()),
        fetch(`/api/frame/${sessionId}/${imf}`).then((r) => r.json()),
      ]);

      stumpX1 = pitchRes.stump_x1;
      stumpX2 = pitchRes.stump_x2;

      // Load images onto canvases
      await loadImageToCanvas(pitchRes.frame_url + "?t=" + Date.now(), canvasPitch, "pitch");
      await loadImageToCanvas(impactRes.frame_url + "?t=" + Date.now(), canvasImpact, "impact");

      // Reset clicks
      pitchClick = null;
      impactClick = null;
      crosshairPitch.style.display = "none";
      crosshairImpact.style.display = "none";
      dotPitch.className = "click-dot click-dot--pending";
      dotImpact.className = "click-dot click-dot--pending";
      pitchCoordsEl.textContent = "not set";
      impactCoordsEl.textContent = "not set";

      // Show frame stage
      showStage("frames");
      showSection(sectionTrajectory);
      btnCompute.disabled = true;

      // Highlight pitch card
      frameCardPitch.classList.add("active-target");
      frameCardImpact.classList.remove("active-target");
      frameInstruction.innerHTML = 'Click on the ball position in the <strong>Pitch</strong> frame';

      setLoading(btnProcessFrames, framesLoader, framesBtnText, false, "", "Extract & Detect Stumps");
    } catch (err) {
      console.error(err);
      alert("Frame processing failed: " + err.message);
      setLoading(btnProcessFrames, framesLoader, framesBtnText, false, "", "Extract & Detect Stumps");
    }
  });

  // ============ Canvas Helpers ============
  function loadImageToCanvas(url, canvas, tag) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        if (tag === "pitch") pitchImgNatural = { w: img.naturalWidth, h: img.naturalHeight };
        if (tag === "impact") impactImgNatural = { w: img.naturalWidth, h: img.naturalHeight };
        resolve();
      };
      img.onerror = reject;
      img.src = url;
    });
  }

  function getImageCoords(e, canvas, naturalSize) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = naturalSize.w / rect.width;
    const scaleY = naturalSize.h / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    return { x, y, displayX: e.clientX - rect.left, displayY: e.clientY - rect.top };
  }

  // ============ Canvas Clicks ============
  canvasPitch.addEventListener("click", (e) => {
    if (!pitchImgNatural) return;
    const coords = getImageCoords(e, canvasPitch, pitchImgNatural);
    pitchClick = { x: coords.x, y: coords.y };
    pitchCoordsEl.textContent = `(${coords.x}, ${coords.y})`;
    dotPitch.className = "click-dot click-dot--done";

    // Show crosshair
    crosshairPitch.style.display = "block";
    crosshairPitch.style.left = coords.displayX + "px";
    crosshairPitch.style.top = coords.displayY + "px";

    // Draw dot on canvas
    const ctx = canvasPitch.getContext("2d");
    ctx.beginPath();
    ctx.arc(coords.x, coords.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#22d3ee";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Move target to impact
    frameCardPitch.classList.remove("active-target");
    frameCardImpact.classList.add("active-target");
    frameInstruction.innerHTML = 'Now click on the ball position in the <strong>Impact</strong> frame';

    checkBothClicked();
  });

  canvasImpact.addEventListener("click", (e) => {
    if (!impactImgNatural) return;
    const coords = getImageCoords(e, canvasImpact, impactImgNatural);
    impactClick = { x: coords.x, y: coords.y };
    impactCoordsEl.textContent = `(${coords.x}, ${coords.y})`;
    dotImpact.className = "click-dot click-dot--done";

    // Show crosshair
    crosshairImpact.style.display = "block";
    crosshairImpact.style.left = coords.displayX + "px";
    crosshairImpact.style.top = coords.displayY + "px";

    // Draw dot on canvas
    const ctx = canvasImpact.getContext("2d");
    ctx.beginPath();
    ctx.arc(coords.x, coords.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#22d3ee";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    frameCardImpact.classList.remove("active-target");
    frameInstruction.innerHTML = '✅ Both points selected — compute trajectory below';

    checkBothClicked();
  });

  function checkBothClicked() {
    if (pitchClick && impactClick) {
      btnCompute.disabled = false;
    }
  }

  // ============ Compute Trajectory ============
  btnCompute.addEventListener("click", async () => {
    if (!pitchClick || !impactClick) return;

    setLoading(btnCompute, computeLoader, computeBtnText, true, "Computing…", "Compute Trajectory & Decision");

    try {
      const res = await fetch(`/api/trajectory/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pitch_x: pitchClick.x,
          pitch_y: pitchClick.y,
          impact_x: impactClick.x,
          impact_y: impactClick.y,
          impact_frame_number: impactFrameNumber,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      // Show trajectory image
      trajectoryImg.src = data.trajectory_url + "?t=" + Date.now();
      showStage("trajectory");

      // Set up Popup
      const isOut = data.decision === "OUT";
      popupTitle.textContent = isOut ? "OUT 🟥" : "NOT OUT 🟩";
      popupTitle.className = "popup-card__title " + (isOut ? "out" : "not-out");
      popupBadge.textContent = isOut ? "OUT" : "NOT OUT";
      popupBadge.style.color = isOut ? "var(--red)" : "var(--green)";
      popupBadge.style.background = isOut ? "var(--red-glow)" : "var(--green-glow)";

      popupDetails.innerHTML = `
        <div class="detail-row">
          <span>Pitch in line</span>
          <span>${data.pitch_in_line ? "✅ Yes" : "❌ No"}</span>
        </div>
        <div class="detail-row">
          <span>Impact in line</span>
          <span>${data.impact_in_line ? "✅ Yes" : "❌ No"}</span>
        </div>
        <div class="detail-row">
          <span>Hitting stumps</span>
          <span>${data.hitting_stumps ? "✅ Yes" : "❌ No"}</span>
        </div>
      `;

      // Sequential DRS Animation
      await runDrsAnimation(data);

      // Show Popup
      decisionPopup.classList.add("popup-overlay--visible");

      setLoading(btnCompute, computeLoader, computeBtnText, false, "", "Compute Trajectory & Decision");

      // --- Play outro video (out.mp4) after decision is shown if OUT ---
      if (isOut) {
        await playOverlayVideo(outroOverlay, outroVideo, "vid/out.mp4");
      }
    } catch (err) {
      console.error(err);
      alert("Trajectory computation failed: " + err.message);
      setLoading(btnCompute, computeLoader, computeBtnText, false, "", "Compute Trajectory & Decision");
    }
  });

  // ============ DRS Animation ============
  function runDrsAnimation(data) {
    return new Promise((resolve) => {
      // Reset
      [drsItemDecision, drsItemWickets, drsItemImpact, drsItemPitching].forEach(item => {
        item.classList.remove('visible');
        item.querySelector('.drs-value').classList.remove('green');
      });
      drsSidebar.style.display = "flex";

      const showItem = (item, valueEl, text, isGreen, delay) => {
        return new Promise(res => {
          setTimeout(() => {
            valueEl.textContent = text;
            if (isGreen) valueEl.classList.add('green');
            item.classList.add('visible');
            res();
          }, delay);
        });
      };

      (async () => {
        // 1. Pitching
        await showItem(drsItemPitching, valPitching, data.pitch_in_line ? "In-Line" : "Outside", data.pitch_in_line, 800);
        // 2. Impact
        await showItem(drsItemImpact, valImpact, data.impact_in_line ? "In-Line" : "Outside", data.impact_in_line, 1200);
        // 3. Wickets
        await showItem(drsItemWickets, valWickets, data.hitting_stumps ? "Hitting" : "Missing", data.hitting_stumps, 1200);
        // 4. Decision
        await showItem(drsItemDecision, valDecision, data.decision, data.decision === "OUT", 1500);
        
        // Wait a bit more before finishing
        setTimeout(resolve, 1500);
      })();
    });
  }

  // ============ Init ============
  function initHub() {
    if (!hubPage || !cardsLbw.length) return;

    cardsLbw.forEach(cardUrl => {
      cardUrl.addEventListener("click", () => {
        // Transition from Hub to LBW Tool
        hubPage.style.opacity = "0";
        setTimeout(() => {
          hubPage.style.display = "none";
          if (landingPage) {
            landingPage.style.display = "flex";
            // Start the LBW landing page logic (audio etc)
            initLandingPage();
          }
        }, 1000);
      });
    });
  }

  function initLandingPage() {
    if (!landingPage || landingPage.style.display === "none") return;

    // Close Popup Logic
    if (btnClosePopup) {
      btnClosePopup.addEventListener("click", () => {
        decisionPopup.classList.remove("popup-overlay--visible");
      });
    }

    // Try to play audio, and add a fallback click listener in case autoplay is blocked by the browser
    let audioPlayed = false;
    const attemptPlayAudio = () => {
      if (!audioPlayed && landingAudio) {
        landingAudio.play().then(() => { audioPlayed = true; }).catch(() => { });
      }
    };
    if (landingAudio) landingAudio.volume = 0.5; // Optional: set initial volume to 50%
    attemptPlayAudio();
    document.body.addEventListener("click", attemptPlayAudio, { once: true });
    document.body.addEventListener("keydown", attemptPlayAudio, { once: true });

    // Simulate model loading delay
    setTimeout(() => {
      if (landingLoader) landingLoader.style.display = "none";
      if (btnEnter) btnEnter.style.display = "flex";
    }, 2500);

    if (btnEnter) {
      btnEnter.addEventListener("click", () => {
        landingPage.classList.add("fade-out");

        // Stop audio when entering the app
        if (landingAudio) {
          landingAudio.pause();
          landingAudio.currentTime = 0;
        }
      });
    }
  }

  showStage("empty");
  initHub();
  // We no longer call initLandingPage here directly, 
  // it's called when 'cardLbw' is clicked.
})();
