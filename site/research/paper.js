(() => {
  "use strict";

  // Preserve bookmarks made before the paper became the default landing page.
  const legacySections = new Set([
    "top", "overview", "method", "results", "rollouts", "reproduce", "scope",
    "motivation", "main", "hero-video", "panel-frozen", "panel-panda", "panel-pilot",
    "tab-frozen", "tab-panda", "tab-pilot", "hero-title", "teaser-caption",
    "deadline-chart-desc", "deadline-chart-title", "site-nav", "method-title",
    "motivation-title", "overview-title", "reproduce-title", "results-title",
    "rollouts-title", "scope-title"
  ]);
  const redirectLegacy = () => {
    if (!legacySections.has(location.hash.slice(1))) return false;
    location.replace(`runtime.html${location.search}${location.hash}`);
    return true;
  };
  if (redirectLegacy()) return;
  window.addEventListener("hashchange", redirectLegacy);

  const video = document.getElementById("film");
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const connection = navigator.connection;
  const transport = document.querySelector(".transport");
  const playback = document.getElementById("playback");
  const scrub = document.getElementById("scrub");
  const timecode = document.getElementById("timecode");

  function updateTransport() {
    const seconds = video.currentTime || 0;
    const formatted = String(seconds.toFixed(2)).padStart(5, "0");
    timecode.textContent = `00:${formatted}`;
    scrub.value = seconds;
    scrub.setAttribute("aria-valuetext", `${seconds.toFixed(2)} seconds`);
    playback.textContent = video.paused ? "▶ Play" : "Ⅱ Pause";
  }
  for (const event of ["play", "pause", "timeupdate", "seeked"]) {
    video.addEventListener(event, updateTransport);
  }
  playback.addEventListener("click", () => {
    if (video.paused) video.play().catch(updateTransport);
    else video.pause();
  });
  scrub.addEventListener("input", () => {
    const target = Number(scrub.value);
    video.pause();
    if (video.readyState >= 1) video.currentTime = target;
    else {
      video.addEventListener("loadedmetadata", () => { video.currentTime = target; }, {once: true});
      video.load();
    }
  });
  transport.hidden = false;
  video.controls = false;
  updateTransport();
  if (!motion.matches && !connection?.saveData) video.play().catch(() => {});
  motion.addEventListener("change", ({matches}) => { if (matches) video.pause(); });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) video.pause();
  });

  const outcomes = {
    44: "Only the fresh requery succeeds on layout 44.",
    45: "Only continuing the queued actions succeeds on layout 45."
  };
  const buttons = [...document.querySelectorAll("[data-layout]")];
  const frames = [...document.querySelectorAll("[data-frame]")];
  for (const button of buttons) {
    button.addEventListener("click", () => {
      const layout = button.dataset.layout;
      for (const other of buttons) other.setAttribute("aria-pressed", String(other === button));
      for (const frame of frames) {
        const [time, view] = frame.dataset.frame.split("-");
        frame.alt = `Layout ${layout} ${view} camera at ${time}`;
        frame.src = `research/layout-${layout}-${time}-${view}.webp`;
      }
      document.getElementById("outcome").textContent = outcomes[layout];
    });
  }
})();
