/**
 * RAP Real-Time Visualizer Client
 * Handles SSE streaming, pan/zoom SVG tree rendering, Blocksworld tabletop,
 * trajectory scrubbing, structured node inspection, and minimal theme switching.
 */

class VisualizerApp {
  constructor() {
    this.treeData = null;
    this.stateData = null;
    this.status = "idle";
    this.iteration = 0;
    this.maxIterations = 50;
    this.goal = null;
    this.initialStateText = "";
    this.selectedNode = null;
    this.bestPathIds = new Set();
    this.bestPathNodes = [];
    this.autoFollow = true;
    this.inspectorMode = "overview";
    this.isStepping = false;
    this.stepStartTime = 0;
    this.stepTicker = null;
    this.syncTimer = null;
    this.eventSource = null;
    this.replayMode = false;
    this.canExitReplay = false;
    this.replayRuns = [];
    this.replayMeta = null;
    this.liveStream = {
      active: false,
      iteration: 0,
      phase: "",
      callId: null,
      reasoning: "",
      content: "",
    };
    this.liveAutoScroll = true;
    this.liveElements = null;

    // Pan & Zoom state
    this.panX = 0;
    this.panY = 0;
    this.zoom = 1;
    this.hasManualViewport = false;
    this.isPanning = false;
    this.panStart = { x: 0, y: 0 };
    this.treeBounds = { width: 400, height: 300 };
    this.nodeElements = new Map();

    this.initElements();
    this.initTheme();
    this.initResizableLayout();
    this.bindEvents();
    this.connectSSE();
    this.fetchInitialData();
  }

  initElements() {
    this.modeBadge = document.getElementById("mode-badge");
    this.sseDot = document.getElementById("sse-dot");
    this.statusDot = document.getElementById("status-dot");
    this.statusText = document.getElementById("status-text");
    this.iterationCounter = document.getElementById("iteration-counter");
    this.progressFill = document.getElementById("progress-fill");

    this.btnStep = document.getElementById("btn-step");
    this.btnPlay = document.getElementById("btn-play");
    this.btnPlayLabel = document.getElementById("btn-play-label");
    this.btnPause = document.getElementById("btn-pause");
    this.btnReset = document.getElementById("btn-reset");
    this.speedSlider = document.getElementById("speed-slider");
    this.speedValue = document.getElementById("speed-value");

    this.btnTheme = document.getElementById("btn-theme");
    this.iconSun = document.getElementById("icon-sun");
    this.iconMoon = document.getElementById("icon-moon");

    this.treeContainer = document.getElementById("tree-container");
    this.treeSvg = document.getElementById("tree-svg");
    this.treeViewport = document.getElementById("tree-viewport");
    this.treeSummary = document.getElementById("tree-summary");
    this.treeTooltip = document.getElementById("tree-tooltip");
    this.btnZoomIn = document.getElementById("btn-zoom-in");
    this.btnZoomOut = document.getElementById("btn-zoom-out");
    this.btnFitTree = document.getElementById("btn-fit-tree");
    this.zoomReadout = document.getElementById("zoom-readout");

    this.goalStrip = document.getElementById("goal-strip");
    this.goalText = document.getElementById("goal-text");
    this.holdingPill = document.getElementById("holding-pill");
    this.stacksArea = document.getElementById("stacks-area");
    this.heldBlockSlot = document.getElementById("held-block-slot");

    this.btnEditProblem = document.getElementById("btn-edit-problem");
    this.problemDialog = document.getElementById("problem-dialog");
    this.problemForm = document.getElementById("problem-form");
    this.problemGoal = document.getElementById("problem-goal");
    this.problemState = document.getElementById("problem-state");
    this.problemError = document.getElementById("problem-error");
    this.btnProblemCancel = document.getElementById("btn-problem-cancel");
    this.btnProblemClose = document.getElementById("btn-problem-close");
    this.btnProblemApply = document.getElementById("btn-problem-apply");

    this.btnReplay = document.getElementById("btn-replay");
    this.replayBadge = document.getElementById("replay-badge");
    this.replayDialog = document.getElementById("replay-dialog");
    this.replaySelect = document.getElementById("replay-run-select");
    this.replayDetails = document.getElementById("replay-details");
    this.replayError = document.getElementById("replay-error");
    this.btnReplayClose = document.getElementById("btn-replay-close");
    this.btnReplayCancel = document.getElementById("btn-replay-cancel");
    this.btnReplayLoad = document.getElementById("btn-replay-load");
    this.btnReplayLive = document.getElementById("btn-replay-live");

    this.trajectoryStrip = document.getElementById("trajectory-strip");
    this.btnFollowBest = document.getElementById("btn-follow-best");
    this.tabOverview = document.getElementById("tab-overview");
    this.tabRaw = document.getElementById("tab-raw");
    this.thoughtModeTag = document.getElementById("thought-mode-tag");
    this.thoughtDisplay = document.getElementById("thought-display");
    this.modelIo = document.getElementById("model-io");
    this.traceSourceBadge = document.getElementById("trace-source-badge");
    this.llmCalls = document.getElementById("llm-calls");
    this.llmCallsBadge = document.getElementById("llm-calls-badge");
    this.nodeIdBadge = document.getElementById("node-id-badge");
    this.terminalBadge = document.getElementById("terminal-badge");
    this.metricsOverview = document.getElementById("metrics-overview");
    this.nodeInspector = document.getElementById("node-inspector");
    this.stateOverview = document.getElementById("state-overview");
    this.stateInspector = document.getElementById("state-inspector");
    this.toastContainer = document.getElementById("toast-container");
  }

  initTheme() {
    const saved = localStorage.getItem("rap-theme") || "dark";
    this.applyTheme(saved);
  }

  applyTheme(theme) {
    const isLight = theme === "light";
    if (isLight) {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    if (this.iconSun && this.iconMoon) {
      this.iconSun.hidden = !isLight;
      this.iconMoon.hidden = isLight;
    }
    localStorage.setItem("rap-theme", isLight ? "light" : "dark");
  }

  bindEvents() {
    if (this.btnStep) this.btnStep.addEventListener("click", () => this.postAction("/api/search/step"));
    if (this.btnPlay) {
      this.btnPlay.addEventListener("click", () => {
        const endpoint = this.status === "paused" ? "/api/search/resume" : "/api/search/start";
        this.postAction(endpoint);
      });
    }
    if (this.btnPause) this.btnPause.addEventListener("click", () => this.postAction("/api/search/pause"));
    if (this.btnReset) {
      this.btnReset.addEventListener("click", () => {
        this.hasManualViewport = false;
        this.postAction("/api/search/reset");
      });
    }

    if (this.btnEditProblem) {
      this.btnEditProblem.addEventListener("click", () => this.openProblemDialog());
    }
    if (this.btnProblemCancel) {
      this.btnProblemCancel.addEventListener("click", () => this.closeProblemDialog());
    }
    if (this.btnProblemClose) {
      this.btnProblemClose.addEventListener("click", () => this.closeProblemDialog());
    }
    if (this.problemForm) {
      this.problemForm.addEventListener("submit", (e) => {
        e.preventDefault();
        this.applyProblem();
      });
    }

    if (this.btnReplay) {
      this.btnReplay.addEventListener("click", () => this.openReplayDialog());
    }
    if (this.btnReplayClose) {
      this.btnReplayClose.addEventListener("click", () => this.closeReplayDialog());
    }
    if (this.btnReplayCancel) {
      this.btnReplayCancel.addEventListener("click", () => this.closeReplayDialog());
    }
    if (this.replaySelect) {
      this.replaySelect.addEventListener("change", () => this.renderReplayDetails());
    }
    if (this.btnReplayLoad) {
      this.btnReplayLoad.addEventListener("click", () => this.loadReplay());
    }
    if (this.btnReplayLive) {
      this.btnReplayLive.addEventListener("click", () => this.exitReplay());
    }

    if (this.speedSlider) {
      let configTimer = null;
      this.speedSlider.addEventListener("input", (e) => {
        const ms = Number(e.target.value);
        if (this.speedValue) this.speedValue.textContent = `${ms} ms`;
        clearTimeout(configTimer);
        configTimer = setTimeout(() => this.updateStepDelay(ms), 120);
      });
    }

    if (this.btnTheme) {
      this.btnTheme.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
        this.applyTheme(current === "light" ? "dark" : "light");
      });
    }

    if (this.btnFollowBest) {
      this.btnFollowBest.addEventListener("click", () => {
        this.setAutoFollow(!this.autoFollow);
        if (this.autoFollow && this.treeData) {
          const target = this.resolveFollowNode();
          if (target) this.selectNode(target, false);
        }
      });
    }

    if (this.tabOverview && this.tabRaw) {
      this.tabOverview.addEventListener("click", () => this.setInspectorMode("overview"));
      this.tabRaw.addEventListener("click", () => this.setInspectorMode("raw"));
    }

    if (this.thoughtDisplay) {
      this.thoughtDisplay.addEventListener("scroll", () => {
        if (!this.liveStream.active) return;
        const dist =
          this.thoughtDisplay.scrollHeight -
          this.thoughtDisplay.scrollTop -
          this.thoughtDisplay.clientHeight;
        this.liveAutoScroll = dist <= 24;
      });
    }

    // Pan & Zoom Controls
    if (this.btnZoomIn) this.btnZoomIn.addEventListener("click", () => this.zoomBy(1.2));
    if (this.btnZoomOut) this.btnZoomOut.addEventListener("click", () => this.zoomBy(1 / 1.2));
    if (this.btnFitTree) {
      this.btnFitTree.addEventListener("click", () => {
        this.hasManualViewport = false;
        this.fitTreeToView();
      });
    }

    this.bindCanvasPanZoom();
    this.bindKeyboardShortcuts();
  }

  bindCanvasPanZoom() {
    if (!this.treeContainer) return;

    this.treeContainer.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || e.target.closest(".canvas-controls") || e.target.closest(".node")) return;
      this.isPanning = true;
      this.hasManualViewport = true;
      this.treeContainer.classList.add("panning");
      this.panStart = { x: e.clientX - this.panX, y: e.clientY - this.panY };
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isPanning) return;
      this.panX = e.clientX - this.panStart.x;
      this.panY = e.clientY - this.panStart.y;
      this.applyViewportTransform();
    });

    window.addEventListener("mouseup", () => {
      if (!this.isPanning) return;
      this.isPanning = false;
      this.treeContainer.classList.remove("panning");
    });

    this.treeContainer.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const rect = this.treeContainer.getBoundingClientRect();
        const cursorX = e.clientX - rect.left;
        const cursorY = e.clientY - rect.top;
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        this.zoomAroundPoint(factor, cursorX, cursorY);
      },
      { passive: false }
    );
  }

  bindKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      const tag = e.target && e.target.tagName ? e.target.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "textarea" || tag === "select") return;

      if (e.code === "Space") {
        e.preventDefault();
        if (this.status === "running") {
          this.postAction("/api/search/pause");
        } else if (this.status !== "finished") {
          const endpoint = this.status === "paused" ? "/api/search/resume" : "/api/search/start";
          this.postAction(endpoint);
        }
      } else if (e.key === "s" || e.key === "S") {
        if (this.status !== "running" && this.status !== "finished") {
          this.postAction("/api/search/step");
        }
      } else if (e.key === "r" || e.key === "R") {
        this.hasManualViewport = false;
        this.postAction("/api/search/reset");
      } else if (e.key === "f" || e.key === "F") {
        this.hasManualViewport = false;
        this.fitTreeToView();
      }
    });
  }

  setAutoFollow(enabled) {
    this.autoFollow = enabled;
    if (this.btnFollowBest) {
      this.btnFollowBest.textContent = `Auto-follow: ${enabled ? "ON" : "OFF"}`;
      this.btnFollowBest.classList.toggle("active", enabled);
      this.btnFollowBest.setAttribute("aria-pressed", String(enabled));
    }
  }

  setInspectorMode(mode) {
    this.inspectorMode = mode;
    const isOverview = mode === "overview";
    if (this.tabOverview) {
      this.tabOverview.classList.toggle("active", isOverview);
      this.tabOverview.setAttribute("aria-selected", String(isOverview));
    }
    if (this.tabRaw) {
      this.tabRaw.classList.toggle("active", !isOverview);
      this.tabRaw.setAttribute("aria-selected", String(!isOverview));
    }
    if (this.metricsOverview) this.metricsOverview.hidden = !isOverview;
    if (this.nodeInspector) this.nodeInspector.hidden = isOverview;
    if (this.stateOverview) this.stateOverview.hidden = !isOverview;
    if (this.stateInspector) this.stateInspector.hidden = isOverview;
  }

  showToast(message) {
    if (!this.toastContainer) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    this.toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 7000);
  }

  async updateStepDelay(ms) {
    try {
      const res = await fetch("/api/search/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ step_delay_ms: ms }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      console.error("Failed to update step delay:", err);
      this.showToast("Could not update step delay.");
    }
  }

  openProblemDialog() {
    if (!this.problemDialog) return;
    this.hideProblemError();
    if (this.problemGoal) {
      this.problemGoal.value = Array.isArray(this.goal) ? this.goal.join("\n") : (this.goal || "");
    }
    if (this.problemState) {
      this.problemState.value = this.initialStateText || "";
    }
    if (typeof this.problemDialog.showModal === "function") {
      this.problemDialog.showModal();
    } else {
      this.problemDialog.setAttribute("open", "");
    }
    if (this.problemGoal) this.problemGoal.focus();
  }

  closeProblemDialog() {
    if (!this.problemDialog) return;
    this.hideProblemError();
    if (typeof this.problemDialog.close === "function") {
      this.problemDialog.close();
    } else {
      this.problemDialog.removeAttribute("open");
    }
  }

  showProblemError(message) {
    if (!this.problemError) return;
    this.problemError.textContent = message;
    this.problemError.hidden = false;
  }

  hideProblemError() {
    if (!this.problemError) return;
    this.problemError.textContent = "";
    this.problemError.hidden = true;
  }

  async applyProblem() {
    if (!this.problemState || !this.problemGoal) return;
    this.hideProblemError();

    const goal = this.problemGoal.value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (goal.length === 0) {
      this.showProblemError("Add at least one goal condition.");
      return;
    }

    if (this.btnProblemApply) this.btnProblemApply.disabled = true;
    try {
      const res = await fetch("/api/problem", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ initial_state: this.problemState.value, goal }),
      });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const errJson = await res.json();
          if (errJson && errJson.detail) {
            detail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
          }
        } catch (_) {}
        throw new Error(detail);
      }

      const data = await res.json();
      this.goal = data.goal || goal;
      this.initialStateText = this.problemState.value;
      this.closeProblemDialog();

      this.liveStream.active = false;
      this.liveElements = null;
      this.selectedNode = null;
      this.hasManualViewport = false;

      if (data.max_iterations !== undefined) this.maxIterations = data.max_iterations;
      if (Array.isArray(data.best_path_ids)) {
        this.bestPathIds = new Set(data.best_path_ids);
      }
      this.renderGoal(this.goal);
      this.updateStatus(data.status, data.iteration);
      if (data.tree) {
        this.treeData = data.tree;
        this.syncTreeAndSelection(true);
      }
      if (data.state) {
        this.stateData = data.state;
        this.renderTabletop();
        this.renderStateOverview(data.state, data.state);
      }
      this.showToast("Problem updated. Press Play to start the new search.");
    } catch (err) {
      console.error("Failed to apply problem:", err);
      this.showProblemError(`Problem rejected: ${err.message}`);
    } finally {
      if (this.btnProblemApply) this.btnProblemApply.disabled = false;
    }
  }

  /* ========================================================================
     Replay Mode (cached runs, no LLM calls)
     ======================================================================== */

  openReplayDialog() {
    this.hideReplayError();
    if (this.replayDialog && !this.replayDialog.open) this.replayDialog.showModal();
    if (this.replaySelect) {
      this.replaySelect.innerHTML = "";
      const loading = document.createElement("option");
      loading.textContent = "Loading saved runs...";
      loading.disabled = true;
      this.replaySelect.appendChild(loading);
    }
    if (this.replayDetails) this.replayDetails.textContent = "";
    if (this.btnReplayLoad) this.btnReplayLoad.disabled = true;
    if (this.btnReplayLive) this.btnReplayLive.hidden = !(this.replayMode && this.canExitReplay);
    this.fetchReplayRuns();
  }

  async fetchReplayRuns() {
    try {
      const res = await fetch("/api/runs?limit=100");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this.replayRuns = Array.isArray(data.runs) ? data.runs : [];
      this.populateReplayRuns();
    } catch (err) {
      this.showReplayError(`Could not load saved runs: ${err.message}`);
    }
  }

  populateReplayRuns() {
    if (!this.replaySelect) return;
    this.replaySelect.innerHTML = "";
    if (this.replayRuns.length === 0) {
      const empty = document.createElement("option");
      empty.textContent = "No saved runs found in the runs directory";
      empty.disabled = true;
      this.replaySelect.appendChild(empty);
      if (this.btnReplayLoad) this.btnReplayLoad.disabled = true;
      if (this.replayDetails) this.replayDetails.textContent = "";
      return;
    }
    for (const run of this.replayRuns) {
      const option = document.createElement("option");
      option.value = String(run.run_id || "");
      const metrics = run.metrics || {};
      const reward = metrics["Best Reward"] !== undefined ? metrics["Best Reward"] : "—";
      option.textContent = `${run.run_id}  ·  ${run.mode || "?"}  ·  ${run.iterations ?? 0} it  ·  reward ${reward}`;
      this.replaySelect.appendChild(option);
    }
    if (this.btnReplayLoad) this.btnReplayLoad.disabled = false;
    this.renderReplayDetails();
  }

  renderReplayDetails() {
    if (!this.replayDetails || !this.replaySelect) return;
    const run = this.replayRuns.find((item) => String(item.run_id) === this.replaySelect.value);
    if (!run) {
      this.replayDetails.textContent = "";
      return;
    }
    const goal = Array.isArray(run.goal) ? run.goal.join(" · ") : (run.goal || "—");
    const model = run.model ? ` · ${run.model}` : "";
    const metrics = run.metrics || {};
    const reward = metrics["Best Reward"] !== undefined ? metrics["Best Reward"] : "—";
    this.replayDetails.textContent = [
      `Backend: ${run.backend || "?"}${model}`,
      `Created: ${run.created_at || "?"}`,
      `Best reward: ${reward}`,
      `Goal: ${goal}`,
    ].join("\n");
  }

  async loadReplay() {
    if (!this.replaySelect || !this.replaySelect.value) return;
    this.hideReplayError();
    const label = this.btnReplayLoad ? this.btnReplayLoad.textContent : "";
    if (this.btnReplayLoad) {
      this.btnReplayLoad.disabled = true;
      this.btnReplayLoad.textContent = "Loading...";
    }
    try {
      const res = await fetch("/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ref: this.replaySelect.value }),
      });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const errJson = await res.json();
          if (errJson && errJson.detail) {
            detail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
          }
        } catch (_) {}
        throw new Error(detail);
      }
      const data = await res.json();
      this.applyReplayPayload(data);
      this.closeReplayDialog();
      const runId = data.run && data.run.run_id ? data.run.run_id : "run";
      this.showToast(`Replay loaded: ${runId}. Use Step or Play.`);
    } catch (err) {
      console.error("Failed to load replay:", err);
      this.showReplayError(`Replay failed: ${err.message}`);
    } finally {
      if (this.btnReplayLoad) {
        this.btnReplayLoad.disabled = false;
        this.btnReplayLoad.textContent = label || "Load replay";
      }
    }
  }

  async exitReplay() {
    this.hideReplayError();
    const label = this.btnReplayLive ? this.btnReplayLive.textContent : "";
    if (this.btnReplayLive) {
      this.btnReplayLive.disabled = true;
      this.btnReplayLive.textContent = "Restoring...";
    }
    try {
      const res = await fetch("/api/replay/exit", { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this.applyReplayPayload(data);
      this.closeReplayDialog();
      this.showToast("Live search restored.");
    } catch (err) {
      console.error("Failed to leave replay:", err);
      this.showReplayError(`Could not restore live search: ${err.message}`);
    } finally {
      if (this.btnReplayLive) {
        this.btnReplayLive.disabled = false;
        this.btnReplayLive.textContent = label || "Back to live";
      }
    }
  }

  applyReplayPayload(data) {
    this.replayMeta = data.run || null;
    this.canExitReplay = Boolean(data.can_exit_replay);
    if (data.max_iterations !== undefined) this.maxIterations = data.max_iterations;
    if (data.goal !== undefined) this.goal = data.goal;
    if (data.mode && this.modeBadge) {
      this.modeBadge.textContent = data.mode;
      this.modeBadge.hidden = false;
    }
    this.bestPathIds = new Set(Array.isArray(data.best_path_ids) ? data.best_path_ids : []);
    this.treeData = data.tree || null;
    this.stateData = data.state || null;
    this.selectedNode = null;
    this.hasManualViewport = false;
    this.liveStream.active = false;
    this.liveElements = null;
    this.renderGoal(this.goal);
    this.setReplayUi(Boolean(data.replay));
    this.updateStatus(data.status, data.iteration);
    if (this.treeData) this.syncTreeAndSelection(true);
    this.renderTabletop();
    this.scheduleTreeFit();
  }

  setReplayUi(enabled) {
    this.replayMode = Boolean(enabled);
    if (this.replayBadge) this.replayBadge.hidden = !this.replayMode;
    if (this.btnEditProblem) this.btnEditProblem.disabled = this.replayMode;
    if (this.btnReplay) this.btnReplay.classList.toggle("active", this.replayMode);
    if (this.btnReplayLive) {
      this.btnReplayLive.hidden = !(this.replayMode && this.canExitReplay);
    }
  }

  showReplayError(message) {
    if (!this.replayError) return;
    this.replayError.textContent = message;
    this.replayError.hidden = false;
  }

  hideReplayError() {
    if (!this.replayError) return;
    this.replayError.textContent = "";
    this.replayError.hidden = true;
  }

  closeReplayDialog() {
    if (this.replayDialog && this.replayDialog.open) this.replayDialog.close();
    this.hideReplayError();
  }

  async postAction(endpoint) {
    const isStep = endpoint.endsWith("/step");
    if (isStep) {
      if (this.isStepping) return;
      this.isStepping = true;
      this.setStepPending(true);
    }
    try {
      const res = await fetch(endpoint, { method: "POST" });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const errJson = await res.json();
          if (errJson && errJson.detail) detail = errJson.detail;
        } catch (_) {}
        throw new Error(detail);
      }
      const data = await res.json();

      if (data.max_iterations !== undefined) {
        this.maxIterations = data.max_iterations;
      }
      if (Array.isArray(data.best_path_ids)) {
        this.bestPathIds = new Set(data.best_path_ids);
      }
      if (data.replay !== undefined) {
        this.setReplayUi(Boolean(data.replay));
      }
      this.updateStatus(data.status, data.iteration);

      if (endpoint.endsWith("/reset")) {
        this.selectedNode = null;
      }
      this.liveStream.active = false;
      this.liveElements = null;
      if (data.tree) {
        this.treeData = data.tree;
        this.syncTreeAndSelection(endpoint.endsWith("/reset"));
      } else {
        await this.fetchTreeAndState();
      }
    } catch (err) {
      console.error(`Error calling ${endpoint}:`, err);
      this.liveStream.active = false;
      this.liveElements = null;
      this.showToast(`Action failed (${endpoint.split("/").pop()}): ${err.message}`);
      await this.fetchTreeAndState();
    } finally {
      if (isStep) {
        this.isStepping = false;
        this.setStepPending(false);
        this.updateStatus(this.status, this.iteration);
      }
    }
  }

  setStepPending(pending) {
    if (this.stepTicker) {
      clearInterval(this.stepTicker);
      this.stepTicker = null;
    }
    if (this.btnStep) {
      this.btnStep.disabled = pending;
      const label = this.btnStep.querySelector("span");
      if (label) {
        if (pending) {
          this.stepStartTime = Date.now();
          label.textContent = "Stepping...";
          this.stepTicker = setInterval(() => {
            const secs = Math.floor((Date.now() - this.stepStartTime) / 1000);
            if (secs >= 1) label.textContent = `Stepping (${secs}s)...`;
          }, 1000);
        } else {
          label.textContent = "Step";
        }
      }
    }
    if (this.btnPlay && pending) {
      this.btnPlay.disabled = true;
    }
    this.updateActiveSyncPolling();
  }

  updateActiveSyncPolling() {
    const needsPoll = this.isStepping || this.status === "running";
    if (needsPoll && !this.syncTimer) {
      this.syncTimer = setInterval(() => this.pollLatestState(), 1200);
    } else if (!needsPoll && this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
  }

  async pollLatestState() {
    try {
      const res = await fetch("/api/meta");
      if (!res.ok) return;
      const meta = await res.json();
      if (meta.iteration !== this.iteration || (meta.status && meta.status.toLowerCase() !== this.status)) {
        this.applyMeta(meta);
        await this.fetchTreeAndState();
      }
    } catch (_) {
      // Ignore transient poll errors while request is in flight
    }
  }

  async fetchInitialData() {
    try {
      const [metaRes, treeRes, stateRes] = await Promise.all([
        fetch("/api/meta"),
        fetch("/api/tree"),
        fetch("/api/state"),
      ]);

      if (metaRes.ok) {
        const meta = await metaRes.json();
        this.applyMeta(meta);
      }
      if (treeRes.ok) {
        this.treeData = await treeRes.json();
      }
      if (stateRes.ok) {
        this.stateData = await stateRes.json();
        this.renderTabletop();
      }
      if (this.treeData) {
        this.syncTreeAndSelection(false);
      }
    } catch (err) {
      console.error("Error fetching initial visualizer data:", err);
      this.showToast("Failed to load initial visualizer state.");
    }
  }

  async fetchTreeAndState() {
    try {
      const [treeRes, stateRes] = await Promise.all([
        fetch("/api/tree"),
        fetch("/api/state"),
      ]);
      if (treeRes.ok) this.treeData = await treeRes.json();
      if (stateRes.ok) {
        this.stateData = await stateRes.json();
        this.renderTabletop();
      }
      if (this.treeData) this.syncTreeAndSelection(false);
    } catch (err) {
      console.error("Error refreshing tree/state:", err);
    }
  }

  applyMeta(meta) {
    if (!meta) return;
    if (meta.max_iterations !== undefined) this.maxIterations = meta.max_iterations;
    if (meta.step_delay_ms !== undefined && this.speedSlider) {
      this.speedSlider.value = String(meta.step_delay_ms);
      if (this.speedValue) this.speedValue.textContent = `${meta.step_delay_ms} ms`;
    }
    if (meta.mode && this.modeBadge) {
      this.modeBadge.textContent = meta.mode;
      this.modeBadge.hidden = false;
    }
    if (Array.isArray(meta.best_path_ids)) {
      this.bestPathIds = new Set(meta.best_path_ids);
    }
    if (meta.goal !== undefined) {
      this.goal = meta.goal;
    }
    if (typeof meta.initial_state === "string") {
      this.initialStateText = meta.initial_state;
    }
    this.replayMeta = meta.run || null;
    this.canExitReplay = Boolean(meta.can_exit_replay);
    this.setReplayUi(Boolean(meta.replay));
    this.renderGoal(meta.goal);
    this.updateStatus(meta.status, meta.iteration);
  }

  renderGoal(goal) {
    if (!this.goalStrip || !this.goalText) return;
    if (!goal || (Array.isArray(goal) && goal.length === 0)) {
      this.goalStrip.hidden = true;
      return;
    }
    let formatted = "";
    if (Array.isArray(goal)) {
      formatted = goal.join(" · ");
    } else if (typeof goal === "object") {
      formatted = Object.entries(goal)
        .map(([top, bottom]) => `${top} on ${bottom}`)
        .join(" · ");
    } else {
      formatted = String(goal);
    }
    this.goalText.textContent = formatted;
    this.goalStrip.hidden = false;
  }

  connectSSE() {
    if (this.eventSource) {
      this.eventSource.close();
    }
    this.setSSEState("connecting");
    this.eventSource = new EventSource("/api/stream");

    this.eventSource.onopen = () => {
      this.setSSEState("connected");
    };

    this.eventSource.onmessage = (event) => {
      this.setSSEState("connected");
      try {
        const payload = JSON.parse(event.data);
        this.handleEvent(payload);
      } catch (err) {
        console.error("SSE parse error:", err);
      }
    };

    this.eventSource.onerror = () => {
      this.setSSEState("disconnected");
    };
  }

  setSSEState(state) {
    if (!this.sseDot) return;
    this.sseDot.className = `sse-dot ${state}`;
    const titles = {
      connected: "Live stream connected",
      connecting: "Connecting to stream...",
      disconnected: "Stream disconnected, reconnecting...",
    };
    this.sseDot.title = titles[state] || state;
  }

  handleEvent(payload) {
    if (payload.max_iterations !== undefined) {
      this.maxIterations = payload.max_iterations;
    }
    if (payload.replay !== undefined) {
      this.setReplayUi(Boolean(payload.replay));
    }
    if (Array.isArray(payload.best_path_ids)) {
      this.bestPathIds = new Set(payload.best_path_ids);
    }
    if (payload.status) {
      this.updateStatus(payload.status, payload.iteration);
    }
    if (payload.goal !== undefined) {
      this.goal = payload.goal;
      this.renderGoal(payload.goal);
    }

    if (payload.type === "error" && payload.message) {
      this.liveStream.active = false;
      this.liveElements = null;
      this.showToast(payload.message);
    }

    if (payload.type === "phase") {
      this.liveStream.active = true;
      if (payload.iteration !== undefined) this.liveStream.iteration = payload.iteration;
      this.liveStream.phase = payload.phase || "";

      if (payload.tree) {
        this.treeData = payload.tree;
        this.renderTree();
        this.renderTrajectoryBar();
      }
      if (payload.preview_state && this.autoFollow) {
        this.stateData = payload.preview_state;
        this.renderTabletop();
        this.renderStateOverview(payload.preview_state, payload.preview_state);
      }

      if (this.autoFollow) {
        this.renderLiveStream();
      }
      return;
    }

    if (payload.type === "token") {
      const incomingCallId = payload.call_id ?? null;
      if (!this.liveStream.active || (incomingCallId !== null && this.liveStream.callId !== incomingCallId)) {
        this.liveStream.active = true;
        this.liveStream.callId = incomingCallId;
        this.liveStream.reasoning = "";
        this.liveStream.content = "";
        this.liveElements = null;
        this.liveAutoScroll = true;
      }
      if (payload.iteration !== undefined) this.liveStream.iteration = payload.iteration;
      if (payload.kind === "reasoning") {
        this.liveStream.reasoning += payload.text || "";
      } else {
        this.liveStream.content += payload.text || "";
      }
      if (this.autoFollow) {
        this.renderLiveStream();
      }
      return;
    }

    const isReset = payload.type === "reset";
    if (isReset || payload.type === "step") {
      this.liveStream.active = false;
      this.liveElements = null;
      if (isReset) {
        this.liveStream.phase = "";
        this.liveStream.callId = null;
        this.liveStream.reasoning = "";
        this.liveStream.content = "";
        this.liveAutoScroll = true;
      }
    }
    if (isReset) {
      this.selectedNode = null;
      this.hasManualViewport = false;
    }

    if (payload.tree) {
      this.treeData = payload.tree;
      this.syncTreeAndSelection(isReset);
    }
  }

  syncTreeAndSelection(isReset) {
    if (!this.treeData) return;

    // Ensure bestPathIds has a valid fallback when backend controller has no mcts attached
    if (this.bestPathIds.size === 0) {
      const fallbackPath = this.computeBestPathFallback(this.treeData);
      this.bestPathIds = new Set(fallbackPath.map((n) => n.id));
    }

    this.renderTree();
    this.renderTrajectoryBar();

    if (isReset) {
      this.selectNode(this.treeData, false);
      return;
    }

    if (this.autoFollow) {
      const target = this.resolveFollowNode();
      if (target) this.selectNode(target, false);
    } else if (this.selectedNode) {
      const refreshed = this.findNodeById(this.treeData, this.selectedNode.id);
      this.selectNode(refreshed || this.treeData, false);
    } else {
      this.selectNode(this.treeData, false);
    }
  }

  updateStatus(status, iteration) {
    this.status = (status || this.status || "idle").toLowerCase();
    this.iteration = iteration !== undefined ? iteration : this.iteration;

    if (this.statusText) this.statusText.textContent = this.status.toUpperCase();
    if (this.iterationCounter) {
      this.iterationCounter.textContent = `${this.iteration} / ${this.maxIterations}`;
    }
    if (this.progressFill) {
      const pct = this.maxIterations > 0
        ? Math.min(100, Math.round((this.iteration / this.maxIterations) * 100))
        : 0;
      this.progressFill.style.width = `${pct}%`;
    }
    if (this.statusDot) this.statusDot.className = `status-dot ${this.status}`;

    const isRunning = this.status === "running";
    const isFinished = this.status === "finished" || this.iteration >= this.maxIterations;

    if (this.btnStep) this.btnStep.disabled = isRunning || isFinished;
    if (this.btnPlay) this.btnPlay.disabled = isRunning || isFinished;
    if (this.btnPause) this.btnPause.disabled = !isRunning;
    if (this.btnPlayLabel) {
      this.btnPlayLabel.textContent = this.status === "paused" ? "Resume" : "Play";
    }
    this.updateActiveSyncPolling();
  }

  computeBestPathFallback(root) {
    const path = [];
    let curr = root;
    while (curr) {
      path.push(curr);
      if (!curr.children || curr.children.length === 0) break;
      const visited = curr.children.filter((c) => (c.visits || 0) > 0);
      if (visited.length === 0) break;
      curr = visited.reduce((best, c) => {
        if ((c.visits || 0) !== (best.visits || 0)) {
          return (c.visits || 0) > (best.visits || 0) ? c : best;
        }
        return (c.Q || 0) > (best.Q || 0) ? c : best;
      }, visited[0]);
    }
    return path;
  }

  collectBestPathNodes(root) {
    const path = [];
    let curr = root;
    while (curr && this.bestPathIds.has(curr.id)) {
      path.push(curr);
      if (!curr.children || curr.children.length === 0) break;
      curr = curr.children.find((c) => this.bestPathIds.has(c.id)) || null;
    }
    return path.length > 0 ? path : this.computeBestPathFallback(root);
  }

  resolveFollowNode() {
    if (!this.treeData) return null;
    const path = this.collectBestPathNodes(this.treeData);
    return path[path.length - 1] || this.treeData;
  }

  findNodeById(root, id) {
    if (!root) return null;
    if (root.id === id) return root;
    if (root.children) {
      for (const child of root.children) {
        const res = this.findNodeById(child, id);
        if (res) return res;
      }
    }
    return null;
  }

  findNodePath(root, id, trail = []) {
    if (!root) return null;
    const nextTrail = [...trail, root];
    if (root.id === id) return nextTrail;
    if (root.children) {
      for (const child of root.children) {
        const res = this.findNodePath(child, id, nextTrail);
        if (res) return res;
      }
    }
    return null;
  }

  /* ========================================================================
     SVG Tree Rendering + Pan & Zoom
     ======================================================================== */

  renderTree() {
    if (!this.treeData || !this.treeViewport) return;

    this.treeViewport.innerHTML = "";
    this.nodeElements.clear();

    const nodes = [];
    const links = [];
    let nextX = 72;

    const assignCoords = (node, depth) => {
      if (!node.children || node.children.length === 0) {
        node.x = nextX;
        nextX += 132;
      } else {
        node.children.forEach((c) => assignCoords(c, depth + 1));
        const first = node.children[0].x;
        const last = node.children[node.children.length - 1].x;
        node.x = (first + last) / 2;
      }
      node.y = 56 + depth * 92;
      nodes.push(node);

      if (node.children) {
        node.children.forEach((c) => links.push({ source: node, target: c }));
      }
    };

    assignCoords(this.treeData, 0);

    const maxDepth = nodes.reduce((m, n) => Math.max(m, n.depth || 0), 0);
    this.treeBounds = {
      width: Math.max(nextX + 40, 220),
      height: Math.max(120 + maxDepth * 92, 220),
    };

    if (this.treeSummary) {
      const plural = nodes.length === 1 ? "node" : "nodes";
      this.treeSummary.textContent = `${nodes.length} ${plural} · depth ${maxDepth}`;
    }

    // Draw Links first (non-best first, best-path on top)
    const sortedLinks = [...links].sort((a, b) => {
      const aBest = this.bestPathIds.has(a.source.id) && this.bestPathIds.has(a.target.id) ? 1 : 0;
      const bBest = this.bestPathIds.has(b.source.id) && this.bestPathIds.has(b.target.id) ? 1 : 0;
      return aBest - bBest;
    });

    sortedLinks.forEach((l) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const midY = (l.source.y + l.target.y) / 2;
      const d = `M ${l.source.x} ${l.source.y} C ${l.source.x} ${midY}, ${l.target.x} ${midY}, ${l.target.x} ${l.target.y}`;
      path.setAttribute("d", d);

      const isBest = this.bestPathIds.has(l.source.id) && this.bestPathIds.has(l.target.id);
      const isVisited = (l.target.visits || 0) > 0;
      const isPreview = Boolean(l.target.is_preview);
      path.setAttribute(
        "class",
        `link ${isPreview ? "preview" : isBest ? "best-path" : isVisited ? "visited" : ""}`
      );
      this.treeViewport.appendChild(path);
    });

    // Draw Nodes
    nodes.forEach((n) => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const isSelected = this.selectedNode && this.selectedNode.id === n.id;
      const isBest = this.bestPathIds.has(n.id);
      const isVisited = (n.visits || 0) > 0;
      const isPreview = Boolean(n.is_preview);

      const classes = ["node"];
      if (isPreview) classes.push("preview");
      if (isVisited) classes.push("visited");
      if (isBest) classes.push("best-path");
      if (n.is_terminal) classes.push("terminal");
      if (isSelected) classes.push("active");

      g.setAttribute("class", classes.join(" "));
      g.setAttribute("transform", `translate(${n.x}, ${n.y})`);

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("r", 20);
      circle.addEventListener("click", (e) => {
        e.stopPropagation();
        this.hideTooltip();
        this.selectNode(n, true);
      });
      circle.addEventListener("mouseenter", (e) => this.showTooltip(n, e));
      circle.addEventListener("mousemove", (e) => this.moveTooltip(e));
      circle.addEventListener("mouseleave", () => this.hideTooltip());
      g.appendChild(circle);

      const visitsText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      visitsText.setAttribute("class", "node-visits");
      visitsText.setAttribute("dy", 4);
      visitsText.textContent = `N:${n.visits}`;
      g.appendChild(visitsText);

      const actionLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      actionLabel.setAttribute("class", "node-action");
      actionLabel.setAttribute("dy", 33);
      actionLabel.textContent = this.formatShortAction(n);
      g.appendChild(actionLabel);

      if (n.visits > 0) {
        const qLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        qLabel.setAttribute("class", "node-q");
        qLabel.setAttribute("dy", 45);
        qLabel.textContent = `Q:${Number(n.Q || 0).toFixed(1)}`;
        g.appendChild(qLabel);
      }

      this.nodeElements.set(n.id, g);
      this.treeViewport.appendChild(g);
    });

    if (!this.hasManualViewport) {
      this.fitTreeToView();
    } else {
      this.applyViewportTransform();
    }
  }

  formatShortAction(node) {
    if (!node.action) return node.depth === 0 ? "Root" : "Step";
    const s = String(node.action);
    const cleaned = s
      .replace(/^BlockAction\(/i, "")
      .replace(/Pick up the ([a-zA-Z0-9_-]+) block\.?/i, "pick $1")
      .replace(/Put down the ([a-zA-Z0-9_-]+) block\.?/i, "drop $1")
      .replace(/Stack the ([a-zA-Z0-9_-]+) block on top of the ([a-zA-Z0-9_-]+) block\.?/i, "stack $1→$2")
      .replace(/Unstack the ([a-zA-Z0-9_-]+) block from on top of the ([a-zA-Z0-9_-]+) block\.?/i, "unstack $1←$2");
    return cleaned.length > 18 ? `${cleaned.slice(0, 17)}…` : cleaned;
  }

  fitTreeToView() {
    if (!this.treeContainer) return;
    const rect = this.treeContainer.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const padding = 36;
    const scaleX = (rect.width - padding) / this.treeBounds.width;
    const scaleY = (rect.height - padding) / this.treeBounds.height;
    this.zoom = Math.max(0.35, Math.min(1.15, Math.min(scaleX, scaleY)));
    this.panX = (rect.width - this.treeBounds.width * this.zoom) / 2;
    this.panY = Math.max(12, (rect.height - this.treeBounds.height * this.zoom) / 2);
    this.applyViewportTransform();
  }

  zoomBy(factor) {
    if (!this.treeContainer) return;
    const rect = this.treeContainer.getBoundingClientRect();
    this.zoomAroundPoint(factor, rect.width / 2, rect.height / 2);
  }

  zoomAroundPoint(factor, cx, cy) {
    const nextZoom = Math.max(0.25, Math.min(2.5, this.zoom * factor));
    const ratio = nextZoom / this.zoom;
    this.panX = cx - (cx - this.panX) * ratio;
    this.panY = cy - (cy - this.panY) * ratio;
    this.zoom = nextZoom;
    this.hasManualViewport = true;
    this.applyViewportTransform();
  }

  applyViewportTransform() {
    if (this.treeViewport) {
      this.treeViewport.setAttribute("transform", `translate(${this.panX}, ${this.panY}) scale(${this.zoom})`);
    }
    if (this.zoomReadout) {
      this.zoomReadout.textContent = `${Math.round(this.zoom * 100)}%`;
    }
  }

  showTooltip(node, event) {
    if (!this.treeTooltip) return;
    const actionText = node.action ? String(node.action) : "Root State";
    const mStr = node.M === null || node.M === undefined ? "—" : Number(node.M).toFixed(2);
    const uctStr = node.uct === null || node.uct === undefined ? "—" : Number(node.uct).toFixed(2);
    this.treeTooltip.innerHTML = `
      <div class="tt-action">${this.escapeHtml(actionText)}</div>
      <div class="tt-metrics">
        <span>N=${node.visits}</span>
        <span>Q=${Number(node.Q || 0).toFixed(2)}</span>
        <span>M=${mStr}</span>
        <span>UCT=${uctStr}</span>
        <span>R=${Number(node.reward || 0).toFixed(1)}</span>
      </div>
    `;
    this.treeTooltip.hidden = false;
    this.moveTooltip(event);
  }

  moveTooltip(event) {
    if (!this.treeTooltip || this.treeTooltip.hidden || !this.treeContainer) return;
    const rect = this.treeContainer.getBoundingClientRect();
    const x = Math.min(rect.width - 200, Math.max(8, event.clientX - rect.left + 12));
    const y = Math.min(rect.height - 70, Math.max(8, event.clientY - rect.top + 12));
    this.treeTooltip.style.left = `${x}px`;
    this.treeTooltip.style.top = `${y}px`;
  }

  hideTooltip() {
    if (this.treeTooltip) this.treeTooltip.hidden = true;
  }

  /* ========================================================================
     Trajectory Breadcrumb & Node Inspector
     ======================================================================== */

  renderTrajectoryBar() {
    if (!this.trajectoryStrip || !this.treeData) return;
    this.bestPathNodes = this.collectBestPathNodes(this.treeData);
    this.trajectoryStrip.innerHTML = "";

    this.bestPathNodes.forEach((node, idx) => {
      if (idx > 0) {
        const sep = document.createElement("span");
        sep.className = "traj-arrow";
        sep.textContent = "›";
        this.trajectoryStrip.appendChild(sep);
      }
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "traj-step";
      if (this.selectedNode && this.selectedNode.id === node.id) {
        btn.classList.add("active");
      }
      btn.textContent = idx === 0 ? "Root" : this.formatShortAction(node);
      btn.title = node.action ? String(node.action) : "Initial state";
      btn.addEventListener("click", () => this.selectNode(node, true));
      this.trajectoryStrip.appendChild(btn);
    });
  }

  selectNode(node, isUserClick = false) {
    if (!node) return;

    if (isUserClick) {
      this.setAutoFollow(false);
      this.liveStream.active = false;
      this.liveElements = null;
    }

    this.selectedNode = node;

    // Immediately sync SVG active ring & trajectory active pill
    this.nodeElements.forEach((el, id) => {
      el.classList.toggle("active", id === node.id);
    });
    if (this.trajectoryStrip) {
      const pills = this.trajectoryStrip.querySelectorAll(".traj-step");
      this.bestPathNodes.forEach((pNode, i) => {
        if (pills[i]) pills[i].classList.toggle("active", pNode.id === node.id);
      });
    }

    this.renderInspector(node);
    this.updateTabletopForNode(node);
  }

  renderLiveStream() {
    if (!this.thoughtDisplay || !this.liveStream.active) return;

    const phaseLabel = this.liveStream.phase ? this.liveStream.phase.toUpperCase() : "LLM";
    const iterLabel = this.liveStream.iteration ? `Iter ${this.liveStream.iteration} · ` : "";
    const callLabel = this.liveStream.callId ? ` · Call #${this.liveStream.callId}` : "";

    if (this.thoughtModeTag) {
      this.thoughtModeTag.textContent = `LIVE · ${iterLabel}${phaseLabel}${callLabel}`;
      this.thoughtModeTag.classList.add("live");
    }

    if (!this.liveElements) {
      this.thoughtDisplay.innerHTML = "";
      const waitingEl = document.createElement("p");
      waitingEl.className = "empty-hint";

      const reasoningLabel = document.createElement("div");
      reasoningLabel.className = "live-section-label";
      reasoningLabel.textContent = "Thinking (Live)";
      reasoningLabel.hidden = true;

      const reasoningPre = document.createElement("pre");
      reasoningPre.className = "live-stream-pre live-reasoning";
      reasoningPre.hidden = true;

      const outputLabel = document.createElement("div");
      outputLabel.className = "live-section-label";
      outputLabel.textContent = "Output (Live)";
      outputLabel.hidden = true;

      const outputPre = document.createElement("pre");
      outputPre.className = "live-stream-pre live-output";
      outputPre.hidden = true;

      this.thoughtDisplay.appendChild(waitingEl);
      this.thoughtDisplay.appendChild(reasoningLabel);
      this.thoughtDisplay.appendChild(reasoningPre);
      this.thoughtDisplay.appendChild(outputLabel);
      this.thoughtDisplay.appendChild(outputPre);

      this.liveElements = {
        waitingEl,
        reasoningLabel,
        reasoningPre,
        outputLabel,
        outputPre,
      };
    }

    const els = this.liveElements;
    const hasReasoning = Boolean(this.liveStream.reasoning);
    const hasContent = Boolean(this.liveStream.content);

    els.waitingEl.hidden = hasReasoning || hasContent;
    if (els.waitingEl.hidden === false) {
      els.waitingEl.innerHTML = `MCTS phase <strong>${this.escapeHtml(phaseLabel)}</strong> — waiting for model stream...`;
    }

    els.reasoningLabel.hidden = !hasReasoning;
    els.reasoningPre.hidden = !hasReasoning;
    if (hasReasoning) {
      els.reasoningPre.textContent = this.liveStream.reasoning;
    }

    els.outputLabel.hidden = !hasContent;
    els.outputPre.hidden = !hasContent;
    if (hasContent) {
      els.outputPre.textContent = this.liveStream.content;
    }

    if (this.liveAutoScroll) {
      this.thoughtDisplay.scrollTop = this.thoughtDisplay.scrollHeight;
    }
  }

  renderInspector(node) {
    if (this.nodeIdBadge) this.nodeIdBadge.textContent = node.id || "—";
    if (this.terminalBadge) this.terminalBadge.hidden = !node.is_terminal;

    const trace = node.llm_trace || null;
    const isFallback = !!(trace && trace.source === "fallback");

    // 1. Thought & Action Card
    if (this.thoughtModeTag) {
      this.thoughtModeTag.classList.remove("live");
      if (isFallback) this.thoughtModeTag.textContent = "Fallback Action";
      else if (node.thought) this.thoughtModeTag.textContent = "Hybrid Thought";
      else if (node.action) this.thoughtModeTag.textContent = "Action";
      else this.thoughtModeTag.textContent = "Root";
    }

    if (this.thoughtDisplay) {
      if (node.thought) {
        this.thoughtDisplay.innerHTML = `
          <div class="thought-quote">${this.escapeHtml(node.thought)}</div>
          ${node.action ? `<div class="action-line"><span>Action</span><span class="action-pill">${this.escapeHtml(node.action)}</span></div>` : ""}
        `;
      } else if (node.action) {
        const hint = isFallback
          ? "No reasoning recorded: the model output was unusable, so the agent enumerated legal actions instead (see Model Input / Output)."
          : "No reasoning recorded for this action (Paper mode or run saved before trace capture).";
        this.thoughtDisplay.innerHTML = `
          <div class="action-line"><span>Action</span><span class="action-pill">${this.escapeHtml(node.action)}</span></div>
          <p class="empty-hint">${hint}</p>
        `;
      } else {
        this.thoughtDisplay.innerHTML = `
          <p class="empty-hint">Initial starting state. Use <strong>Step</strong> or <strong>Play</strong> to expand the reasoning tree.</p>
        `;
      }
    }

    // 1b. Model Input / Output Card
    if (this.traceSourceBadge) {
      this.traceSourceBadge.hidden = !trace;
      this.traceSourceBadge.classList.toggle("danger", isFallback);
      if (trace) {
        const sourceLabels = {
          model: "model call",
          fallback: "fallback",
          "model-paper": "paper call",
        };
        this.traceSourceBadge.textContent = sourceLabels[trace.source] || trace.source || "trace";
      }
    }

    if (this.modelIo) {
      if (!trace) {
        this.modelIo.innerHTML = `<p class="empty-hint">${
          node.action
            ? "No model trace saved for this node (run created before trace capture was enabled)."
            : "Root node: no model call was made."
        }</p>`;
      } else {
        const reason = trace.reason
          ? `<p class="trace-warning">${this.escapeHtml(trace.reason)}</p>`
          : "";
        const modelMeta = trace.model ? `<span>model: ${this.escapeHtml(trace.model)}</span>` : "";
        const logprobMeta =
          trace.logprob !== undefined
            ? `<span>logprob: ${Number(trace.logprob).toFixed(3)}</span>`
            : "";
        const probMeta =
          trace.prob !== undefined
            ? `<span>r1: ${Number(trace.prob).toFixed(3)}</span>`
            : "";
        const isPaper = trace.source === "model-paper" || trace.source === "rollout";
        const outputTitle = isPaper ? "Scored action · r1" : "Model output · raw";
        const outputText = isPaper
          ? trace.source === "rollout"
            ? "(rollout: action likelihood skipped, only the goal heuristic r2 is used)"
            : `log P(action | state, goal) = ${Number(trace.logprob ?? 0).toFixed(3)}\n` +
              `normalized r1 = ${Number(trace.prob ?? 0).toFixed(3)}\n\n` +
              "Paper mode enumerates every legal action and ranks it; it does not generate a single plan line."
          : trace.response || "(empty)";
        const proposed = trace.proposed
          ? `<details class="io-block"><summary>Parsed proposal</summary><pre class="io-pre">${this.escapeHtml(
              JSON.stringify(trace.proposed, null, 2)
            )}</pre></details>`
          : "";
        this.modelIo.innerHTML = `
          ${reason}
          <div class="trace-meta">
            <span>source: ${this.escapeHtml(trace.source || "model")}</span>
            ${modelMeta}
            ${logprobMeta}
            ${probMeta}
          </div>
          <details class="io-block" open>
            <summary>${outputTitle}</summary>
            <pre class="io-pre">${this.escapeHtml(outputText)}</pre>
          </details>
          <details class="io-block">
            <summary>Model input · prompt</summary>
            <pre class="io-pre">${this.escapeHtml(trace.prompt || "(not recorded)")}</pre>
          </details>
          ${proposed}
        `;
      }
    }

    // 1c. All LLM calls made while expanding this node, plus the whole path
    const llmCalls = Array.isArray(node.llm_calls) ? node.llm_calls : [];
    const pathNodes = this.treeData ? this.findNodePath(this.treeData, node.id) || [] : [];
    const pathGroups = [];
    pathNodes.forEach((pathNode) => {
      const calls = Array.isArray(pathNode.llm_calls) ? pathNode.llm_calls : [];
      if (calls.length) {
        pathGroups.push({
          depth: pathNode.depth,
          action: pathNode.action,
          calls,
        });
      }
    });
    const pathCallCount = pathGroups.reduce((total, group) => total + group.calls.length, 0);

    const renderCall = (call, idx, openDefault = false) => {
      const phase =
        call.phase === "world-model"
          ? "world model"
          : call.phase === "action-scoring"
            ? "action scoring"
            : call.phase === "self-evaluation"
              ? "self-evaluation"
              : "model call";
      const actionLabel = call.action ? ` · ${call.action}` : "";
      const meta = [];
      if (call.logprob !== undefined && call.logprob !== null) {
        meta.push(`logprob ${Number(call.logprob).toFixed(3)}`);
      }
      const r1 = call.r1 ?? call.prob;
      if (r1 !== undefined && r1 !== null) {
        meta.push(`r1 ${Number(r1).toFixed(3)}`);
      }
      if (call.r2 !== undefined && call.r2 !== null) meta.push(`r2 ${Number(call.r2).toFixed(3)}`);
      if (call.r3 !== undefined && call.r3 !== null) meta.push(`r3 ${Number(call.r3).toFixed(3)}`);
      if (call.p_yes !== undefined && call.p_no !== undefined) {
        meta.push(`P(Yes) ${Number(call.p_yes).toFixed(3)} · P(No) ${Number(call.p_no).toFixed(3)}`);
      }
      if (call.source) meta.push(`source ${call.source}`);
      if (call.error) meta.push("error");
      const metaHtml = meta.length
        ? `<div class="trace-meta">${meta
            .map((m) => `<span>${this.escapeHtml(m)}</span>`)
            .join("")}</div>`
        : "";
      const errorHtml = call.error
        ? `<p class="trace-warning">${this.escapeHtml(call.error)}</p>`
        : "";
      const responseLabel = call.phase === "action-scoring" ? "Scored continuation" : "Response";
      return `<details class="io-block call-block${call.error ? " call-error" : ""}" ${openDefault ? "open" : ""}>
        <summary>#${idx + 1} ${this.escapeHtml(phase)}${this.escapeHtml(actionLabel)}</summary>
        ${metaHtml}
        ${errorHtml}
        <div class="live-section-label">Prompt</div>
        <pre class="io-pre">${this.escapeHtml(call.prompt || "(not recorded)")}</pre>
        <div class="live-section-label">${responseLabel}</div>
        <pre class="io-pre">${this.escapeHtml(call.response || "(not recorded)")}</pre>
      </details>`;
    };

    if (this.llmCallsBadge) {
      this.llmCallsBadge.hidden = pathCallCount === 0;
      if (pathCallCount) {
        this.llmCallsBadge.textContent = `${pathCallCount} call${pathCallCount === 1 ? "" : "s"}`;
      }
    }

    if (this.llmCalls) {
      const stepHtml = llmCalls.length
        ? llmCalls.map((call, idx) => renderCall(call, idx, idx === 0)).join("")
        : `<p class="empty-hint">${
            node.action
              ? "No calls were made while expanding this node (leaf, terminal, or run saved before call capture)."
              : "Root node: no calls were made while expanding it."
          }</p>`;

      const pathHtml = pathGroups.length
        ? `<details class="io-block path-calls"${llmCalls.length ? "" : " open"}>
            <summary>Whole path · ${pathCallCount} call${pathCallCount === 1 ? "" : "s"} across ${pathGroups.length} step${pathGroups.length === 1 ? "" : "s"}</summary>
            ${pathGroups
              .map(
                (group) => `<div class="path-call-group">
                  <div class="live-section-label">Depth ${group.depth}${group.action ? ` · after ${this.escapeHtml(this.formatShortAction(group))}` : ""}</div>
                  ${group.calls.map((call, idx) => renderCall(call, idx)).join("")}
                </div>`
              )
              .join("")}
          </details>`
        : "";

      this.llmCalls.innerHTML = `
        <div class="live-section-label">This step · ${llmCalls.length} call${llmCalls.length === 1 ? "" : "s"}</div>
        ${stepHtml}
        ${pathHtml}
      `;
    }

    // 2. Metrics Overview + Raw JSON
    if (this.metricsOverview) {
      const fmt = (v, digits = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(digits));
      const items = [
        ["Visits", String(node.visits ?? 0)],
        ["Mean Q", fmt(node.Q)],
        ["Max M", fmt(node.M)],
        ["Reward", fmt(node.reward, 1)],
        ["UCT", fmt(node.uct)],
        ["Depth", String(node.depth ?? 0)],
        ["Terminal", node.is_terminal ? "Yes" : "No"],
        ["Children", String((node.children || []).length)],
      ];
      this.metricsOverview.innerHTML = items
        .map(
          ([label, val]) => `
            <div class="kpi-cell">
              <span class="kpi-label">${label}</span>
              <span class="kpi-val">${this.escapeHtml(val)}</span>
            </div>
          `
        )
        .join("");
    }

    if (this.nodeInspector) {
      this.nodeInspector.textContent = JSON.stringify(
        {
          id: node.id,
          action: node.action,
          thought: node.thought,
          llm_trace: node.llm_trace,
          llm_calls: node.llm_calls,
          visits: node.visits,
          Q: node.Q,
          M: node.M,
          reward: node.reward,
          uct: node.uct,
          is_terminal: node.is_terminal,
          depth: node.depth,
        },
        null,
        2
      );
    }

    // 3. Physical State Overview + Raw JSON
    if (this.stateInspector) {
      this.stateInspector.textContent =
        typeof node.state === "object" && node.state !== null
          ? JSON.stringify(node.state, null, 2)
          : String(node.state ?? "");
    }
  }

  renderStateOverview(parsedState, rawState) {
    if (!this.stateOverview) return;
    if (parsedState && (Array.isArray(parsedState.stacks) || parsedState.holding !== undefined)) {
      const stacksText =
        parsedState.stacks && parsedState.stacks.length > 0
          ? parsedState.stacks.map((s) => `[${s.join(" → ")}]`).join("  ")
          : "None";
      const clearText =
        parsedState.clear && parsedState.clear.length > 0 ? parsedState.clear.join(", ") : "—";
      const holdingText = parsedState.holding || "Empty";

      this.stateOverview.innerHTML = `
        <div class="state-row"><span class="state-key">Holding</span><span class="state-val">${this.escapeHtml(holdingText)}</span></div>
        <div class="state-row"><span class="state-key">Stacks</span><span class="state-val">${this.escapeHtml(stacksText)}</span></div>
        <div class="state-row"><span class="state-key">Clear</span><span class="state-val">${this.escapeHtml(clearText)}</span></div>
      `;
    } else {
      this.stateOverview.innerHTML = `
        <div class="state-row"><span class="state-key">State</span><span class="state-val">${this.escapeHtml(String(rawState ?? "—"))}</span></div>
      `;
    }
  }

  /* ========================================================================
     Blocksworld Tabletop Simulator
     ======================================================================== */

  parseTextState(text) {
    if (!text || typeof text !== "string") return null;
    const holdingMatch = text.match(/(?:hand\s+is\s+holding\s+the|holding\s+the)\s+([a-zA-Z0-9_-]+)\s+block/i);
    const holding = holdingMatch ? holdingMatch[1].toLowerCase() : null;

    const onMatches = [
      ...text.matchAll(/the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+on\s+top\s+of\s+the\s+([a-zA-Z0-9_-]+)\s+block/gi),
    ];
    const onMap = {};
    onMatches.forEach((m) => {
      onMap[m[2].toLowerCase()] = m[1].toLowerCase();
    });

    const tableMatches = [
      ...text.matchAll(/the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+on\s+the\s+table/gi),
    ];
    const tableBlocks = tableMatches.map((m) => m[1].toLowerCase()).sort();

    const clearMatches = [...text.matchAll(/the\s+([a-zA-Z0-9_-]+)\s+block\s+is\s+clear/gi)];
    const clear = clearMatches.map((m) => m[1].toLowerCase()).sort();

    if (!holding && tableBlocks.length === 0 && onMatches.length === 0) {
      return null;
    }

    const stacks = [];
    tableBlocks.forEach((bottom) => {
      const stack = [bottom];
      let curr = bottom;
      while (onMap[curr]) {
        curr = onMap[curr];
        stack.push(curr);
      }
      stacks.push(stack);
    });

    return { stacks, holding, clear };
  }

  updateTabletopForNode(node) {
    let parsed = null;
    if (node && node.state) {
      if (typeof node.state === "object" && Array.isArray(node.state.stacks)) {
        parsed = node.state;
        this.stateData = node.state;
      } else if (typeof node.state === "string") {
        parsed = this.parseTextState(node.state);
        if (parsed) this.stateData = parsed;
      }
    }
    this.renderStateOverview(parsed || this.stateData, node ? node.state : "");
    this.renderTabletop();
  }

  renderTabletop() {
    if (!this.stacksArea || !this.heldBlockSlot) return;
    this.stacksArea.innerHTML = "";
    this.heldBlockSlot.innerHTML = "";

    const data = this.stateData || {};
    const holding = data.holding || null;
    const stacks = Array.isArray(data.stacks) ? data.stacks : [];

    if (this.holdingPill) {
      this.holdingPill.textContent = holding ? `Holding: ${holding}` : "Hand: empty";
    }

    if (holding) {
      const blockEl = document.createElement("div");
      blockEl.className = `block ${holding.toLowerCase()}`;
      blockEl.textContent = holding;
      this.heldBlockSlot.appendChild(blockEl);
    }

    if (stacks.length === 0 && !holding) {
      const hint = document.createElement("div");
      hint.className = "empty-table-hint";
      hint.textContent = "No block configuration in current state.";
      this.stacksArea.appendChild(hint);
      return;
    }

    stacks.forEach((stack) => {
      const stackEl = document.createElement("div");
      stackEl.className = "block-stack";
      stack.forEach((blockName) => {
        const blockEl = document.createElement("div");
        blockEl.className = `block ${blockName.toLowerCase()}`;
        blockEl.textContent = blockName;
        stackEl.appendChild(blockEl);
      });
      this.stacksArea.appendChild(stackEl);
    });
  }

  /* ========================================================================
     Resizable Layout (draggable splitters)
     ======================================================================== */

  layoutStorageKey() {
    return "rap.layout.v1";
  }

  loadLayout() {
    try {
      const raw = localStorage.getItem(this.layoutStorageKey());
      return raw ? JSON.parse(raw) || {} : {};
    } catch {
      return {};
    }
  }

  saveLayout(patch) {
    this.layoutState = { ...(this.layoutState || {}), ...patch };
    try {
      localStorage.setItem(this.layoutStorageKey(), JSON.stringify(this.layoutState));
    } catch {
      /* localStorage can be unavailable (private mode); layout still works in memory */
    }
  }

  applyLayout() {
    const layout = this.layoutState || {};
    const workspace = document.querySelector(".workspace");
    if (workspace && Number.isFinite(layout.splitTop)) {
      workspace.style.setProperty("--split-top", `${layout.splitTop}%`);
    }

    const applyGrows = (selector, values) => {
      const items = Array.from(document.querySelectorAll(selector));
      if (!Array.isArray(values) || values.length !== items.length) return;
      items.forEach((el, index) => {
        const value = Number(values[index]);
        if (Number.isFinite(value) && value > 0) el.style.flexGrow = String(value);
      });
    };

    applyGrows(".workspace-top > .tree-panel, .workspace-top > .tabletop-panel", layout.topGrows);
    applyGrows(".inspector-content > .inspector-card", layout.cardGrows);
  }

  initResizableLayout() {
    this.layoutState = this.loadLayout();
    this.applyLayout();

    const topItems = [
      document.querySelector(".workspace-top > .tree-panel"),
      document.querySelector(".workspace-top > .tabletop-panel"),
    ];
    this.setupColumnSplitter(
      document.getElementById("splitter-tree-tabletop"),
      topItems,
      0,
      "topGrows",
      [240, 240]
    );

    const cards = Array.from(document.querySelectorAll(".inspector-content > .inspector-card"));
    Array.from(document.querySelectorAll("[data-card-splitter]")).forEach((splitter, index) => {
      this.setupColumnSplitter(splitter, cards, index, "cardGrows", [150, 150]);
    });

    this.setupRowSplitter(
      document.getElementById("splitter-top-inspector"),
      document.querySelector(".workspace"),
      document.querySelector(".workspace-top"),
      document.querySelector(".inspector-panel")
    );
  }

  scheduleTreeFit() {
    if (this.treeFitFrame) cancelAnimationFrame(this.treeFitFrame);
    this.treeFitFrame = requestAnimationFrame(() => {
      this.treeFitFrame = null;
      if (!this.treeData || this.hasManualViewport) return;
      this.fitTreeToView();
    });
  }

  setupColumnSplitter(splitter, items, index, groupKey, mins) {
    if (!splitter || !Array.isArray(items) || index >= items.length - 1) return;
    const prev = items[index];
    const next = items[index + 1];
    if (!prev || !next) return;

    const [minPrev, minNext] = mins || [140, 140];

    // Convert every sibling's flex-grow to its current pixel width, so the
    // ratios stay proportional when only two of them are dragged.
    const normalize = () => {
      const widths = items.map((el) => Math.max(1, el.getBoundingClientRect().width));
      items.forEach((el, i) => {
        el.style.flexGrow = String(widths[i]);
        el.style.flexShrink = "1";
      });
      return widths;
    };

    const persist = () => {
      this.saveLayout({
        [groupKey]: items.map((el) => Number(el.style.flexGrow) || 1),
      });
    };

    const applyPair = (widths, dx) => {
      const total = widths[index] + widths[index + 1];
      const newPrev = Math.min(total - minNext, Math.max(minPrev, widths[index] + dx));
      prev.style.flexGrow = String(newPrev);
      next.style.flexGrow = String(total - newPrev);
      return newPrev - widths[index];
    };

    let drag = null;
    splitter.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      splitter.setPointerCapture?.(event.pointerId);
      drag = { startX: event.clientX, widths: normalize() };
      document.body.classList.add("is-resizing", "is-resizing-col");
    });

    splitter.addEventListener("pointermove", (event) => {
      if (!drag) return;
      applyPair(drag.widths, event.clientX - drag.startX);
    });

    const finish = () => {
      if (!drag) return;
      drag = null;
      document.body.classList.remove("is-resizing", "is-resizing-col");
      persist();
      this.scheduleTreeFit();
    };
    splitter.addEventListener("pointerup", finish);
    splitter.addEventListener("pointercancel", finish);

    splitter.addEventListener("dblclick", () => {
      items.forEach((el) => {
        el.style.flexGrow = "";
        el.style.flexShrink = "";
      });
      this.saveLayout({ [groupKey]: null });
      this.scheduleTreeFit();
    });

    splitter.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const widths = normalize();
      applyPair(widths, event.key === "ArrowLeft" ? -16 : 16);
      persist();
      this.scheduleTreeFit();
    });
  }

  setupRowSplitter(splitter, workspace, top, inspector) {
    if (!splitter || !workspace || !top || !inspector) return;
    const MIN_TOP = 220;
    const MIN_BOTTOM = 150;

    const apply = (clientY, offset) => {
      const topRect = top.getBoundingClientRect();
      const inspectorRect = inspector.getBoundingClientRect();
      const splitterRect = splitter.getBoundingClientRect();
      const total = topRect.height + inspectorRect.height + splitterRect.height;
      if (total <= 0) return null;
      const desired = clientY - offset - topRect.top;
      const newTop = Math.min(total - MIN_BOTTOM, Math.max(MIN_TOP, desired));
      const pct = (newTop / total) * 100;
      workspace.style.setProperty("--split-top", `${pct.toFixed(3)}%`);
      return pct;
    };

    let drag = null;
    splitter.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      splitter.setPointerCapture?.(event.pointerId);
      const rect = splitter.getBoundingClientRect();
      drag = { offset: event.clientY - rect.top };
      document.body.classList.add("is-resizing", "is-resizing-row");
    });

    splitter.addEventListener("pointermove", (event) => {
      if (!drag) return;
      apply(event.clientY, drag.offset);
    });

    const persist = () => {
      const raw = workspace.style.getPropertyValue("--split-top").replace("%", "").trim();
      const pct = Number(raw);
      this.saveLayout({ splitTop: Number.isFinite(pct) ? pct : null });
      this.scheduleTreeFit();
    };

    const finish = () => {
      if (!drag) return;
      drag = null;
      document.body.classList.remove("is-resizing", "is-resizing-row");
      persist();
    };
    splitter.addEventListener("pointerup", finish);
    splitter.addEventListener("pointercancel", finish);

    splitter.addEventListener("dblclick", () => {
      workspace.style.removeProperty("--split-top");
      this.saveLayout({ splitTop: null });
      this.scheduleTreeFit();
    });

    splitter.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
      event.preventDefault();
      const rect = splitter.getBoundingClientRect();
      apply(rect.top + rect.height / 2 + (event.key === "ArrowUp" ? -16 : 16), rect.height / 2);
      persist();
    });
  }

  escapeHtml(str) {
    return String(str).replace(/[&<>'"]/g, (tag) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;",
    }[tag] || tag));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new VisualizerApp();
});
