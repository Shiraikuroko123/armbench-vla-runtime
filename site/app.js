/* ArmBench site runtime. Keep this file dependency-free so the GitHub Pages
 * build has the same behavior when opened from a local static server. */
(function () {
  "use strict";

  var root = document.documentElement;
  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  var saveData = Boolean(connection && connection.saveData);
  var activeLanguage = "zh";
  var pageTitles = Object.freeze({
    zh: "ArmBench | pi0.5 运行时证据",
    en: "ArmBench | pi0.5 Runtime Evidence"
  });

  // The visible figures are duplicated here as a small content contract. If a
  // future artifact changes, one verified source can update the metric strip.
  var evidence = Object.freeze({
    liveResponses: "35",
    policyLatency: "82.75 / 89.56",
    inferenceTicks: "290 / 311",
    liveViolations: "0",
    violations: "0",
    g02Success: "38 / 40",
    g02OverlapTicks: "4,521"
  });

  Object.keys(evidence).forEach(function (key) {
    document.querySelectorAll('[data-evidence="' + key + '"]').forEach(function (element) {
      element.textContent = evidence[key];
    });
  });

  function readStoredLanguage() {
    try {
      var saved = window.localStorage.getItem("armbench-language");
      return saved === "en" || saved === "zh" ? saved : null;
    } catch (error) {
      return null;
    }
  }

  function storeLanguage(language) {
    try {
      window.localStorage.setItem("armbench-language", language);
    } catch (error) {
      // Private browsing can deny storage; the current session still works.
    }
  }

  function updateLocalizedAttributes(language) {
    document.querySelectorAll("[data-aria-zh][data-aria-en]").forEach(function (element) {
      element.setAttribute("aria-label", language === "zh" ? element.dataset.ariaZh : element.dataset.ariaEn);
    });
    document.querySelectorAll("[data-title-zh][data-title-en]").forEach(function (element) {
      element.setAttribute("title", language === "zh" ? element.dataset.titleZh : element.dataset.titleEn);
    });
  }

  function setLanguage(language, persist) {
    activeLanguage = language === "en" ? "en" : "zh";
    root.dataset.lang = activeLanguage;
    root.lang = activeLanguage === "zh" ? "zh-CN" : "en";
    document.title = pageTitles[activeLanguage];
    document.querySelectorAll("[data-set-lang]").forEach(function (button) {
      button.setAttribute("aria-pressed", button.dataset.setLang === activeLanguage ? "true" : "false");
    });
    updateLocalizedAttributes(activeLanguage);
    if (persist) storeLanguage(activeLanguage);
    document.dispatchEvent(new CustomEvent("armbench:language", { detail: { language: activeLanguage } }));
  }

  var browserLanguage = (navigator.language || "").toLowerCase();
  setLanguage(readStoredLanguage() || (browserLanguage.indexOf("zh") === 0 ? "zh" : "en"), false);
  document.querySelectorAll("[data-set-lang]").forEach(function (button) {
    button.addEventListener("click", function () {
      setLanguage(button.dataset.setLang, true);
    });
  });

  // The mobile navigation is a real stateful menu, rather than CSS-only hover.
  var menuButton = document.querySelector("[data-menu-button]");
  var navigation = document.getElementById("site-nav");

  function syncMenuLabel() {
    if (!menuButton) return;
    var open = menuButton.getAttribute("aria-expanded") === "true";
    var label = open
      ? (activeLanguage === "zh" ? menuButton.dataset.ariaCloseZh : menuButton.dataset.ariaCloseEn)
      : (activeLanguage === "zh" ? menuButton.dataset.ariaOpenZh : menuButton.dataset.ariaOpenEn);
    menuButton.setAttribute("aria-label", label);
  }

  function setMenu(open, returnFocus) {
    if (!menuButton || !navigation) return;
    menuButton.setAttribute("aria-expanded", open ? "true" : "false");
    navigation.dataset.open = open ? "true" : "false";
    document.body.classList.toggle("menu-open", open);
    syncMenuLabel();
    if (!open && returnFocus) menuButton.focus();
  }

  if (menuButton && navigation) {
    menuButton.addEventListener("click", function () {
      setMenu(menuButton.getAttribute("aria-expanded") !== "true", false);
    });
    navigation.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () { setMenu(false, false); });
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") setMenu(false, true);
    });
    document.addEventListener("pointerdown", function (event) {
      if (menuButton.getAttribute("aria-expanded") !== "true") return;
      if (!navigation.contains(event.target) && !menuButton.contains(event.target)) setMenu(false, false);
    });
    var desktopMedia = window.matchMedia("(min-width: 901px)");
    var closeDesktopMenu = function (event) { if (event.matches) setMenu(false, false); };
    if (typeof desktopMedia.addEventListener === "function") desktopMedia.addEventListener("change", closeDesktopMenu);
    else desktopMedia.addListener(closeDesktopMenu);
    document.addEventListener("armbench:language", syncMenuLabel);
    syncMenuLabel();
  }

  // Attach a video source only when it is useful. Posters remain the no-data
  // fallback, which keeps the initial GitHub Pages request small.
  var videoPromises = new WeakMap();

  function showVideoError(video) {
    var frame = video.closest(".comparison-media, .hero");
    var fallback = frame && frame.querySelector("[data-video-fallback]");
    if (fallback) fallback.hidden = false;
  }

  function attachVideo(video) {
    if (!video || videoPromises.has(video)) return videoPromises.get(video);
    var promise = new Promise(function (resolve) {
      var sources = Array.from(video.querySelectorAll("source[data-src]"));
      if (!sources.length) { resolve(video); return; }
      var settled = false;
      var finish = function () { if (!settled) { settled = true; resolve(video); } };
      var fail = function () { showVideoError(video); finish(); };
      video.addEventListener("loadedmetadata", finish, { once: true });
      video.addEventListener("error", fail, { once: true });
      sources.forEach(function (source) {
        source.src = source.dataset.src;
        source.removeAttribute("data-src");
        source.addEventListener("error", fail, { once: true });
      });
      video.load();
      // Some browsers do not emit metadata for a blocked local file.
      window.setTimeout(finish, 3000);
    });
    videoPromises.set(video, promise);
    return promise;
  }

  var heroVideo = document.querySelector("[data-hero-video]");
  var heroToggle = heroVideo && document.querySelector('[data-video-toggle="' + heroVideo.id + '"]');
  var heroUserPaused = reducedMotion.matches || saveData;

  function syncHeroButton() {
    if (!heroToggle || !heroVideo) return;
    var playing = !heroVideo.paused && !heroVideo.ended;
    var label = playing
      ? (activeLanguage === "zh" ? heroToggle.dataset.ariaPauseZh : heroToggle.dataset.ariaPauseEn)
      : (activeLanguage === "zh" ? heroToggle.dataset.ariaPlayZh : heroToggle.dataset.ariaPlayEn);
    heroToggle.dataset.playing = playing ? "true" : "false";
    heroToggle.setAttribute("aria-pressed", playing ? "true" : "false");
    heroToggle.setAttribute("aria-label", label);
    heroToggle.setAttribute("title", label);
  }

  function playHero(manual) {
    if (!heroVideo) return;
    if (manual) heroUserPaused = false;
    attachVideo(heroVideo).then(function () {
      return heroVideo.play();
    }).catch(function () {
      // Autoplay may be denied; the poster and manual button remain usable.
    }).finally(syncHeroButton);
  }

  function pauseHero(manual) {
    if (!heroVideo) return;
    if (manual) heroUserPaused = true;
    heroVideo.pause();
    syncHeroButton();
  }

  if (heroVideo) {
    heroVideo.addEventListener("play", syncHeroButton);
    heroVideo.addEventListener("pause", syncHeroButton);
    heroVideo.addEventListener("ended", syncHeroButton);
    heroVideo.addEventListener("error", function () { showVideoError(heroVideo); });
    if (heroToggle) heroToggle.addEventListener("click", function () {
      if (heroVideo.paused) playHero(true); else pauseHero(true);
    });
    document.addEventListener("armbench:language", syncHeroButton);
    syncHeroButton();
    if (!heroUserPaused) window.setTimeout(function () { playHero(false); }, 250);
  }

  var lazyVideos = Array.from(document.querySelectorAll("[data-lazy-video]"));
  lazyVideos.forEach(function (video) {
    var loadOnIntent = function () { attachVideo(video); };
    video.addEventListener("pointerdown", loadOnIntent, { once: true });
    video.addEventListener("keydown", loadOnIntent, { once: true });
  });
  if ("IntersectionObserver" in window) {
    var mediaObserver = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        attachVideo(entry.target);
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "240px 0px", threshold: 0.01 });
    lazyVideos.forEach(function (video) { mediaObserver.observe(video); });
  }

  // Matched π0.5 clips share one timeline. This is intentionally a viewer for
  // the separate LIBERO experiment, not a claim that it drives the Panda path.
  var comparisonVideos = Array.from(document.querySelectorAll("[data-sync-video]"));
  var comparisonPlay = document.querySelector("[data-comparison-play]");
  var comparisonRestart = document.querySelector("[data-comparison-restart]");
  var comparisonScrubber = document.querySelector("[data-comparison-scrubber]");
  var comparisonTime = document.querySelector("[data-comparison-time]");
  var comparisonSpeed = document.querySelector("[data-comparison-speed]");
  var comparisonPlaying = false;
  var comparisonFrame = 0;

  function finiteDuration(video) {
    return video && Number.isFinite(video.duration) ? video.duration : 22;
  }

  function comparisonDuration() {
    return comparisonVideos.reduce(function (max, video) { return Math.max(max, finiteDuration(video)); }, 22);
  }

  function formatTime(seconds) {
    var safe = Math.max(0, Number(seconds) || 0);
    return Math.floor(safe / 60) + ":" + String(Math.floor(safe % 60)).padStart(2, "0");
  }

  function currentComparisonTime() {
    return comparisonVideos.length ? (Number(comparisonVideos[0].currentTime) || 0) : 0;
  }

  function updateComparisonButton() {
    if (!comparisonPlay) return;
    comparisonPlay.dataset.playing = comparisonPlaying ? "true" : "false";
    comparisonPlay.setAttribute("aria-pressed", comparisonPlaying ? "true" : "false");
    comparisonPlay.setAttribute("aria-label", comparisonPlaying
      ? (activeLanguage === "zh" ? "暂停两段视频" : "Pause both videos")
      : (activeLanguage === "zh" ? "播放两段视频" : "Play both videos"));
  }

  function renderComparisonProgress() {
    var time = currentComparisonTime();
    var duration = comparisonDuration();
    if (comparisonScrubber) comparisonScrubber.value = String(Math.min(100, duration ? (time / duration) * 100 : 0));
    if (comparisonTime) comparisonTime.textContent = formatTime(time) + " / " + formatTime(duration);
  }

  function syncComparisonFollowers() {
    if (!comparisonVideos.length) return;
    var master = comparisonVideos[0];
    comparisonVideos.slice(1).forEach(function (video) {
      if (!Number.isFinite(video.duration)) return;

      // The successful clip is shorter than the failed rollout. Keep it on its
      // last decodable frame instead of seeking to duration, which some
      // browsers interpret as "ended" and reset to frame zero.
      var lastFrame = Math.max(0, video.duration - (1 / 30));
      var targetTime = Math.min(master.currentTime, lastFrame);
      if (Math.abs(video.currentTime - targetTime) > 0.08) {
        try { video.currentTime = targetTime; } catch (error) { /* unloaded */ }
      }
      if (master.currentTime >= lastFrame && !video.paused) video.pause();
    });
    renderComparisonProgress();
    if (comparisonPlaying) comparisonFrame = window.requestAnimationFrame(syncComparisonFollowers);
  }

  function stopComparisonFrame() {
    if (comparisonFrame) window.cancelAnimationFrame(comparisonFrame);
    comparisonFrame = 0;
  }

  function ensureComparisonLoaded() {
    return Promise.all(comparisonVideos.map(attachVideo));
  }

  function setComparisonTime(time) {
    var safe = Math.max(0, Number(time) || 0);
    comparisonVideos.forEach(function (video) {
      if (Number.isFinite(video.duration)) {
        var lastFrame = Math.max(0, video.duration - (1 / 30));
        video.currentTime = Math.min(safe, lastFrame);
      }
    });
    renderComparisonProgress();
  }

  function pauseComparison() {
    comparisonPlaying = false;
    stopComparisonFrame();
    comparisonVideos.forEach(function (video) { video.pause(); });
    updateComparisonButton();
  }

  function playComparison() {
    if (!comparisonVideos.length) return;
    ensureComparisonLoaded().then(function () {
      comparisonVideos.forEach(function (video) {
        video.playbackRate = comparisonSpeed ? Number(comparisonSpeed.value) : 1;
      });
      return Promise.all(comparisonVideos.map(function (video) { return video.play(); }));
    }).then(function () {
      comparisonPlaying = true;
      updateComparisonButton();
      stopComparisonFrame();
      comparisonFrame = window.requestAnimationFrame(syncComparisonFollowers);
    }).catch(function () {
      comparisonVideos.forEach(showVideoError);
      pauseComparison();
    });
  }

  if (comparisonPlay) comparisonPlay.addEventListener("click", function () {
    if (comparisonPlaying) pauseComparison(); else playComparison();
  });
  if (comparisonRestart) comparisonRestart.addEventListener("click", function () {
    ensureComparisonLoaded().then(function () {
      pauseComparison();
      setComparisonTime(0);
    });
  });
  if (comparisonScrubber) comparisonScrubber.addEventListener("input", function () {
    ensureComparisonLoaded().then(function () { setComparisonTime((Number(comparisonScrubber.value) / 100) * comparisonDuration()); });
  });
  if (comparisonSpeed) comparisonSpeed.addEventListener("change", function () {
    comparisonVideos.forEach(function (video) { video.playbackRate = Number(comparisonSpeed.value); });
  });
  comparisonVideos.forEach(function (video) {
    video.addEventListener("timeupdate", renderComparisonProgress);
    video.addEventListener("ended", function () { if (video === comparisonVideos[0]) pauseComparison(); });
    video.addEventListener("error", function () { showVideoError(video); });
  });
  document.addEventListener("armbench:language", updateComparisonButton);
  updateComparisonButton();
  renderComparisonProgress();

  // Copy buttons use the system clipboard when available and a textarea
  // fallback for the local-file / older-browser case.
  function fallbackCopy(value) {
    var input = document.createElement("textarea");
    input.value = value;
    input.setAttribute("readonly", "");
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.appendChild(input);
    input.select();
    var copied = document.execCommand("copy");
    input.remove();
    return copied;
  }

  function copyText(value) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(value);
    return fallbackCopy(value) ? Promise.resolve() : Promise.reject(new Error("copy failed"));
  }

  function renderCopyButton(button, copied) {
    if (copied) {
      button.textContent = activeLanguage === "zh" ? "已复制" : "COPIED";
      return;
    }
    button.innerHTML = activeLanguage === "zh" ? "复制" : "Copy";
  }

  document.querySelectorAll("[data-copy]").forEach(function (button) {
    var resetTimer = 0;
    renderCopyButton(button, false);
    button.addEventListener("click", function () {
      window.clearTimeout(resetTimer);
      copyText(button.dataset.copy).then(function () {
        button.classList.add("copied");
        renderCopyButton(button, true);
        resetTimer = window.setTimeout(function () {
          button.classList.remove("copied");
          renderCopyButton(button, false);
        }, 1800);
      }).catch(function () {
        button.textContent = activeLanguage === "zh" ? "复制失败" : "Failed";
      });
    });
    document.addEventListener("armbench:language", function () {
      if (!button.classList.contains("copied")) renderCopyButton(button, false);
    });
  });

  // Reveal is intentionally a small enhancement: all content remains visible
  // without JavaScript, while an observer adds a quiet entrance on long pages.
  var revealItems = Array.from(document.querySelectorAll(".reveal"));
  if ("IntersectionObserver" in window && !reducedMotion.matches) {
    var revealObserver = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8%", threshold: 0.05 });
    revealItems.forEach(function (item) { revealObserver.observe(item); });
  } else {
    revealItems.forEach(function (item) { item.classList.add("is-visible"); });
  }

  // Highlight the section currently under the reading line in the header.
  var sectionLinks = Array.from(document.querySelectorAll(".nav-links a[href^='#']"));
  if ("IntersectionObserver" in window) {
    var sectionObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        sectionLinks.forEach(function (link) {
          var active = link.getAttribute("href") === "#" + entry.target.id;
          if (active) link.setAttribute("aria-current", "location"); else link.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-38% 0px -55%", threshold: 0 });
    sectionLinks.forEach(function (link) {
      var section = document.querySelector(link.getAttribute("href"));
      if (section) sectionObserver.observe(section);
    });
  }

  function handleReducedMotionChange(event) {
    if (!event.matches) return;
    pauseHero(false);
    pauseComparison();
    revealItems.forEach(function (item) { item.classList.add("is-visible"); });
  }

  if (typeof reducedMotion.addEventListener === "function") reducedMotion.addEventListener("change", handleReducedMotionChange);
  else reducedMotion.addListener(handleReducedMotionChange);

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      pauseHero(false);
      pauseComparison();
    } else if (!heroUserPaused && !reducedMotion.matches && !saveData) {
      playHero(false);
    }
  });
})();
