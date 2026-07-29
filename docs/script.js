const cases = {
  atmospheric_pressure: {
    concept: "ATMOSPHERIC PRESSURE",
    title: "A sealed plastic bottle collapses.",
    prompt: "A sealed plastic bottle collapsing as air is extracted.",
    baseline: "assets/videos/atmospheric_pressure-baseline.mp4",
    draft: "assets/videos/atmospheric_pressure-draft.mp4",
    ours: "assets/videos/atmospheric_pressure-ours.mp4",
  },
  buoyancy: {
    concept: "BUOYANCY",
    title: "A wooden toy reaches the waterline.",
    prompt: "A wooden toy is placed gently on the surface of a bowl of water.",
    baseline: "assets/videos/buoyancy-baseline.mp4",
    draft: "assets/videos/buoyancy-draft.mp4",
    ours: "assets/videos/buoyancy-ours.mp4",
  },
  hardness: {
    concept: "MATERIAL HARDNESS",
    title: "An egg shatters on impact.",
    prompt: "An egg is thrown forcefully at a rough rock surface and shatters on impact.",
    baseline: "assets/videos/hardness-baseline.mp4",
    draft: "assets/videos/hardness-draft.mp4",
    ours: "assets/videos/hardness-ours.mp4",
  },
  sublimation: {
    concept: "SUBLIMATION",
    title: "Dry ice sublimates as temperature rises.",
    prompt: "Dry ice sublimating directly as temperature rises.",
    baseline: "assets/videos/sublimation-baseline.mp4",
    draft: "assets/videos/sublimation-draft.mp4",
    ours: "assets/videos/sublimation-ours.mp4",
  },
};

const state = { currentCase: "atmospheric_pressure", syncing: false, playbackRequest: 0 };
const $ = (selector) => document.querySelector(selector);
const baselineVideo = $("#baselineVideo");
const draftVideo = $("#draftVideo");
const oursVideo = $("#oursVideo");
const playButton = $("#playButton");
const demoVideos = [baselineVideo, draftVideo, oursVideo];

function renderIcons() {
  if (window.lucide) {
    window.lucide.createIcons({ attrs: { "stroke-width": 1.9 } });
  }
}

function formatTime(value) {
  if (!Number.isFinite(value)) return "00:00";
  const minutes = Math.floor(value / 60).toString().padStart(2, "0");
  const seconds = Math.floor(value % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function setPlaybackButton(isPlaying) {
  playButton.innerHTML = isPlaying
    ? "<span aria-hidden='true'>&#10074;&#10074;</span>"
    : "<span aria-hidden='true'>&#9654;</span>";
  const label = isPlaying ? "Pause videos" : "Play videos";
  playButton.setAttribute("aria-label", label);
  playButton.title = label;
}

function pauseVideos() {
  state.playbackRequest += 1;
  demoVideos.forEach((video) => video.pause());
  setPlaybackButton(false);
}

function loadCase(key) {
  const data = cases[key];
  pauseVideos();
  state.currentCase = key;
  $("#caseConcept").textContent = data.concept;
  $("#caseTitle").textContent = data.title;
  $("#casePrompt").textContent = data.prompt;
  $("#caseNumber").textContent = `${Object.keys(cases).indexOf(key) + 1}`.padStart(2, "0") + " / 04";
  baselineVideo.src = data.baseline;
  draftVideo.src = data.draft;
  oursVideo.src = data.ours;
  demoVideos.forEach((video) => {
    video.load();
    video.currentTime = 0;
  });
  updateTimecode();
}

function updateTimecode() {
  $("#timeCurrent").textContent = formatTime(oursVideo.currentTime);
  $("#timeTotal").textContent = formatTime(oursVideo.duration);
}

function syncVideos(source) {
  if (state.syncing) return;
  state.syncing = true;
  const time = source.currentTime;
  demoVideos.forEach((video) => {
    if (video !== source && Math.abs(video.currentTime - time) > 0.08) video.currentTime = time;
  });
  state.syncing = false;
  updateTimecode();
}

function togglePlay() {
  if (demoVideos.some((video) => !video.paused)) {
    pauseVideos();
    return;
  }

  const request = ++state.playbackRequest;
  Promise.all(demoVideos.map((video) => video.play()))
    .then(() => {
      if (request === state.playbackRequest) setPlaybackButton(true);
    })
    .catch(() => {
      if (request === state.playbackRequest) pauseVideos();
    });
}

document.querySelectorAll(".case-tab").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".case-tab").forEach((tab) => {
    tab.classList.remove("is-active");
    tab.setAttribute("aria-selected", "false");
  });
  button.classList.add("is-active");
  button.setAttribute("aria-selected", "true");
  loadCase(button.dataset.case);
}));

playButton.addEventListener("click", togglePlay);
demoVideos.forEach((video) => {
  video.addEventListener("timeupdate", () => syncVideos(video));
  video.addEventListener("loadedmetadata", updateTimecode);
  video.addEventListener("play", () => {
    if (demoVideos.every((item) => !item.paused)) setPlaybackButton(true);
  });
  video.addEventListener("pause", () => {
    if (demoVideos.some((item) => item.paused)) setPlaybackButton(false);
  });
});

$("#copyCitation").addEventListener("click", async () => {
  const button = $("#copyCitation");
  try {
    await navigator.clipboard.writeText($("#bibtex").textContent);
    button.innerHTML = "<i data-lucide='check' aria-hidden='true'></i><span>Copied</span>";
    renderIcons();
    setTimeout(() => {
      button.innerHTML = "<i data-lucide='copy' aria-hidden='true'></i><span>Copy BibTeX</span>";
      renderIcons();
    }, 1800);
  } catch {
    button.textContent = "Select BibTeX";
  }
});

loadCase(state.currentCase);
renderIcons();
