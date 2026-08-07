(function () {
  "use strict";

  var root = document.documentElement;
  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  var saveData = Boolean(connection && connection.saveData);
  var activeLanguage = "zh";

  root.dataset.dataSaving = saveData ? "true" : "false";

  function readStoredLanguage() {
    try {
      return window.localStorage.getItem("armbench-language");
    } catch (error) {
      return null;
    }
  }

  function storeLanguage(language) {
    try {
      window.localStorage.setItem("armbench-language", language);
    } catch (error) {
      // A blocked storage API must not block the language control.
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
    document.title = activeLanguage === "zh"
      ? "ArmBench | VLA 运行时监督"
      : "ArmBench | Runtime supervision for action-chunk VLA";

    document.querySelectorAll("[data-set-lang]").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.setLang === activeLanguage));
    });

    updateLocalizedAttributes(activeLanguage);
    if (persist) storeLanguage(activeLanguage);
    document.dispatchEvent(new CustomEvent("armbench:language", { detail: activeLanguage }));
  }

  document.querySelectorAll("[data-set-lang]").forEach(function (button) {
    button.addEventListener("click", function () {
      setLanguage(button.dataset.setLang, true);
    });
  });

  var storedLanguage = readStoredLanguage();
  var browserPrefersChinese = (navigator.language || "").toLowerCase().startsWith("zh");
  setLanguage(storedLanguage === "zh" || storedLanguage === "en" ? storedLanguage : (browserPrefersChinese ? "zh" : "en"), false);

  var menuButton = document.querySelector("[data-menu-button]");
  var navigation = document.getElementById("site-nav");
  var menuBackground = Array.from(document.querySelectorAll(".skip-link, main, .site-footer"));

  function setMenuBackgroundInert(inert) {
    menuBackground.forEach(function (element) {
      element.inert = inert;
    });
  }

  function visibleHeaderControls() {
    return Array.from(document.querySelectorAll(".site-header a, .site-header button")).filter(function (element) {
      return !element.disabled && element.getClientRects().length > 0;
    });
  }

  function syncMenuLabel() {
    if (!menuButton) return;
    var open = menuButton.getAttribute("aria-expanded") === "true";
    menuButton.setAttribute("aria-label", activeLanguage === "zh"
      ? (open ? "关闭导航" : "打开导航")
      : (open ? "Close navigation" : "Open navigation"));
  }

  function setMenu(open, returnFocus) {
    if (!menuButton || !navigation) return;
    menuButton.setAttribute("aria-expanded", String(open));
    navigation.classList.toggle("open", open);
    document.body.classList.toggle("menu-open", open);
    setMenuBackgroundInert(open);
    syncMenuLabel();

    if (open) {
      var firstLink = navigation.querySelector("a");
      if (firstLink) firstLink.focus();
    } else if (returnFocus) {
      menuButton.focus();
    }
  }

  if (menuButton && navigation) {
    menuButton.addEventListener("click", function () {
      setMenu(menuButton.getAttribute("aria-expanded") !== "true", false);
    });

    navigation.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        var wasOpen = menuButton.getAttribute("aria-expanded") === "true";
        setMenu(false, wasOpen);
      });
    });

    document.addEventListener("keydown", function (event) {
      var menuOpen = menuButton.getAttribute("aria-expanded") === "true";
      if (event.key === "Escape" && menuOpen) {
        setMenu(false, true);
        return;
      }

      if (event.key === "Tab" && menuOpen) {
        var controls = visibleHeaderControls();
        var firstControl = controls[0];
        var lastControl = controls[controls.length - 1];
        if (!firstControl || !lastControl) return;

        if (event.shiftKey && document.activeElement === firstControl) {
          event.preventDefault();
          lastControl.focus();
        } else if (!event.shiftKey && document.activeElement === lastControl) {
          event.preventDefault();
          firstControl.focus();
        }
      }
    });

    document.addEventListener("pointerdown", function (event) {
      if (menuButton.getAttribute("aria-expanded") !== "true") return;
      if (navigation.contains(event.target) || menuButton.contains(event.target)) return;
      setMenu(false, false);
    });

    var desktopNavigation = window.matchMedia("(min-width: 861px)");
    var closeMenuAtDesktop = function (event) {
      if (event.matches) setMenu(false, false);
    };
    if (typeof desktopNavigation.addEventListener === "function") {
      desktopNavigation.addEventListener("change", closeMenuAtDesktop);
    } else {
      desktopNavigation.addListener(closeMenuAtDesktop);
    }
  }

  document.addEventListener("armbench:language", syncMenuLabel);
  syncMenuLabel();

  var resultTabs = Array.from(document.querySelectorAll("[data-result-tab]"));
  var resultPanels = Array.from(document.querySelectorAll("[data-result-panel]"));

  function activateResultTab(tab, moveFocus) {
    if (!tab) return;
    resultTabs.forEach(function (candidate) {
      var selected = candidate === tab;
      candidate.setAttribute("aria-selected", String(selected));
      candidate.tabIndex = selected ? 0 : -1;
    });
    resultPanels.forEach(function (panel) {
      panel.hidden = panel.id !== tab.dataset.resultTab;
    });
    if (tab.dataset.resultTab !== "result-measured") pauseComparison();
    if (moveFocus) tab.focus();
  }

  resultTabs.forEach(function (tab, index) {
    tab.addEventListener("click", function () {
      activateResultTab(tab, false);
    });
    tab.addEventListener("keydown", function (event) {
      var nextIndex = index;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % resultTabs.length;
      if (event.key === "ArrowLeft") nextIndex = (index - 1 + resultTabs.length) % resultTabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = resultTabs.length - 1;
      if (nextIndex !== index) {
        event.preventDefault();
        activateResultTab(resultTabs[nextIndex], true);
      }
    });
  });

  var videoPromises = new WeakMap();

  function showVideoError(video) {
    var frame = video.closest(".video-frame, .testbed-media");
    var fallback = frame ? frame.querySelector("[data-video-fallback]") : null;
    if (fallback) fallback.hidden = false;
    video.dataset.mediaError = "true";
  }

  function attachVideo(video) {
    if (!video) return Promise.reject(new Error("Missing video element"));
    if (video.readyState >= 1) return Promise.resolve(video);
    if (videoPromises.has(video)) return videoPromises.get(video);

    var promise = new Promise(function (resolve, reject) {
      var settled = false;
      var finish = function () {
        if (settled) return;
        settled = true;
        resolve(video);
      };
      var fail = function () {
        if (settled) return;
        settled = true;
        showVideoError(video);
        reject(new Error("Video failed to load: " + (video.currentSrc || video.id || "unknown")));
      };

      video.addEventListener("loadedmetadata", finish, { once: true });
      video.addEventListener("error", fail, { once: true });

      var attachedSource = false;
      video.querySelectorAll("source[data-src]").forEach(function (source) {
        source.addEventListener("error", fail, { once: true });
        source.src = source.dataset.src;
        source.removeAttribute("data-src");
        attachedSource = true;
      });

      if (attachedSource || video.currentSrc) {
        video.dataset.sourceAttached = "true";
        video.load();
      } else {
        fail();
      }
    });

    videoPromises.set(video, promise);
    return promise;
  }

  var lazyVideos = Array.from(document.querySelectorAll("[data-lazy-video]"));

  lazyVideos.forEach(function (video) {
    var loadOnIntent = function () {
      attachVideo(video).catch(function () {});
    };
    video.addEventListener("pointerdown", loadOnIntent, { once: true });
    video.addEventListener("keydown", loadOnIntent, { once: true });
  });

  if (!saveData && "IntersectionObserver" in window) {
    var mediaObserver = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        attachVideo(entry.target).catch(function () {});
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "320px 0px" });
    lazyVideos.forEach(function (video) {
      mediaObserver.observe(video);
    });
  } else if (!saveData) {
    lazyVideos.forEach(function (video) {
      attachVideo(video).catch(function () {});
    });
  }

  var heroVideo = document.querySelector("[data-hero-video]");
  var heroToggle = heroVideo ? document.querySelector("[data-video-toggle='" + heroVideo.id + "']") : null;
  var heroUserPaused = reducedMotion.matches || saveData;
  var heroInView = true;

  function syncHeroButton() {
    if (!heroToggle || !heroVideo) return;
    var playing = !heroVideo.paused && !heroVideo.ended;
    heroToggle.dataset.playing = String(playing);
    heroToggle.setAttribute("aria-pressed", String(playing));
    var label = activeLanguage === "zh"
      ? (playing ? heroToggle.dataset.ariaPauseZh : heroToggle.dataset.ariaPlayZh)
      : (playing ? heroToggle.dataset.ariaPauseEn : heroToggle.dataset.ariaPlayEn);
    heroToggle.setAttribute("aria-label", label);
    heroToggle.setAttribute("title", label);
  }

  function playHero(manual) {
    if (!heroVideo) return Promise.resolve();
    if (manual) heroUserPaused = false;
    return attachVideo(heroVideo).then(function () {
      if (!heroInView || document.hidden) return;
      return heroVideo.play();
    }).catch(function () {
      heroUserPaused = true;
      syncHeroButton();
    });
  }

  function pauseHero(manual) {
    if (!heroVideo) return;
    if (manual) heroUserPaused = true;
    heroVideo.pause();
    syncHeroButton();
  }

  if (heroVideo && heroToggle) {
    heroToggle.addEventListener("click", function () {
      if (heroVideo.paused || heroVideo.ended) {
        playHero(true);
      } else {
        pauseHero(true);
      }
    });
    heroVideo.addEventListener("play", syncHeroButton);
    heroVideo.addEventListener("pause", syncHeroButton);
    heroVideo.addEventListener("ended", syncHeroButton);

    if ("IntersectionObserver" in window) {
      var heroObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          heroInView = entry.isIntersecting && entry.intersectionRatio > 0.1;
          if (!heroInView) {
            pauseHero(false);
          } else if (!heroUserPaused && !reducedMotion.matches && !saveData) {
            playHero(false);
          }
        });
      }, { threshold: [0, 0.1, 0.5] });
      heroObserver.observe(heroVideo);
    }

    if (!heroUserPaused) playHero(false);
  }

  document.addEventListener("armbench:language", syncHeroButton);

  var comparisonVideos = Array.from(document.querySelectorAll("[data-sync-video]"));
  var comparisonPlay = document.querySelector("[data-comparison-play]");
  var comparisonRestart = document.querySelector("[data-comparison-restart]");
  var comparisonScrubber = document.querySelector("[data-comparison-scrubber]");
  var comparisonTime = document.querySelector("[data-comparison-time]");
  var comparisonSpeed = document.querySelector("[data-comparison-speed]");
  var comparisonPlaying = false;
  var comparisonFrame = 0;

  function finiteDuration(video) {
    return Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
  }

  function comparisonDuration() {
    var duration = comparisonVideos.reduce(function (maximum, video) {
      return Math.max(maximum, finiteDuration(video));
    }, 0);
    return duration || 22;
  }

  function comparisonMaster() {
    return comparisonVideos.reduce(function (selected, video) {
      return !selected || finiteDuration(video) > finiteDuration(selected) ? video : selected;
    }, null);
  }

  function formatTime(seconds) {
    var value = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
    var minutes = Math.floor(value / 60);
    var remainder = Math.floor(value % 60);
    return minutes + ":" + String(remainder).padStart(2, "0");
  }

  function currentComparisonTime() {
    var master = comparisonMaster();
    return master && Number.isFinite(master.currentTime) ? master.currentTime : 0;
  }

  function updateComparisonButton() {
    if (!comparisonPlay) return;
    comparisonPlay.dataset.playing = String(comparisonPlaying);
    var label = activeLanguage === "zh"
      ? (comparisonPlaying ? "暂停两段" : "播放两段")
      : (comparisonPlaying ? "Pause both" : "Play both");
    var text = comparisonPlay.querySelector("[data-comparison-play-label]");
    if (text) text.textContent = label;
    var ariaLabel = activeLanguage === "zh"
      ? (comparisonPlaying ? comparisonPlay.dataset.ariaPauseZh : comparisonPlay.dataset.ariaPlayZh)
      : (comparisonPlaying ? comparisonPlay.dataset.ariaPauseEn : comparisonPlay.dataset.ariaPlayEn);
    comparisonPlay.setAttribute("aria-label", ariaLabel);
  }

  function renderComparisonProgress() {
    var duration = comparisonDuration();
    var current = Math.min(currentComparisonTime(), duration);
    if (comparisonScrubber) comparisonScrubber.value = String(duration > 0 ? (current / duration) * 100 : 0);
    if (comparisonTime) comparisonTime.textContent = formatTime(current) + " / " + formatTime(duration);
  }

  function syncComparisonFollowers() {
    if (!comparisonPlaying) return;
    var master = comparisonMaster();
    if (!master) return;
    var time = master.currentTime;

    comparisonVideos.forEach(function (video) {
      if (video === master) return;
      var duration = finiteDuration(video);
      if (!duration) return;
      if (time >= duration - 0.03) {
        if (video.currentTime < duration - 0.08) video.currentTime = Math.max(0, duration - 0.03);
        if (!video.paused) video.pause();
      } else {
        if (Math.abs(video.currentTime - time) > 0.18) video.currentTime = time;
        if (video.paused && !video.dataset.syncStarting) {
          video.dataset.syncStarting = "true";
          video.play().catch(function () {}).finally(function () {
            delete video.dataset.syncStarting;
          });
        }
      }
    });

    renderComparisonProgress();
    if (master.ended || master.currentTime >= finiteDuration(master) - 0.03) {
      pauseComparison();
      renderComparisonProgress();
      return;
    }
    comparisonFrame = window.requestAnimationFrame(syncComparisonFollowers);
  }

  function ensureComparisonLoaded() {
    if (comparisonPlay) comparisonPlay.setAttribute("aria-busy", "true");
    return Promise.all(comparisonVideos.map(function (video) {
      return attachVideo(video);
    })).then(function (videos) {
      if (comparisonPlay) comparisonPlay.removeAttribute("aria-busy");
      renderComparisonProgress();
      return videos;
    }).catch(function (error) {
      if (comparisonPlay) {
        comparisonPlay.removeAttribute("aria-busy");
        comparisonPlay.disabled = true;
      }
      throw error;
    });
  }

  function setComparisonTime(time) {
    var duration = comparisonDuration();
    var target = Math.max(0, Math.min(time, duration));
    comparisonVideos.forEach(function (video) {
      var videoDuration = finiteDuration(video);
      if (!videoDuration) return;
      video.currentTime = Math.min(target, Math.max(0, videoDuration - 0.03));
    });
    renderComparisonProgress();
  }

  function pauseComparison() {
    comparisonPlaying = false;
    if (comparisonFrame) window.cancelAnimationFrame(comparisonFrame);
    comparisonFrame = 0;
    comparisonVideos.forEach(function (video) {
      video.pause();
    });
    updateComparisonButton();
  }

  function playComparison() {
    if (!comparisonVideos.length) return;
    ensureComparisonLoaded().then(function () {
      var duration = comparisonDuration();
      if (currentComparisonTime() >= duration - 0.06) setComparisonTime(0);
      var time = currentComparisonTime();
      var rate = comparisonSpeed ? Number(comparisonSpeed.value) : 1;
      var playRequests = [];

      comparisonVideos.forEach(function (video) {
        video.playbackRate = rate;
        var videoDuration = finiteDuration(video);
        video.currentTime = Math.min(time, Math.max(0, videoDuration - 0.03));
        if (time < videoDuration - 0.03) playRequests.push(video.play());
      });

      return Promise.allSettled(playRequests).then(function (results) {
        comparisonPlaying = results.some(function (result) {
          return result.status === "fulfilled";
        });
        updateComparisonButton();
        if (comparisonPlaying) {
          if (comparisonFrame) window.cancelAnimationFrame(comparisonFrame);
          comparisonFrame = window.requestAnimationFrame(syncComparisonFollowers);
        }
      });
    }).catch(function () {
      comparisonPlaying = false;
      updateComparisonButton();
    });
  }

  if (comparisonPlay) {
    comparisonPlay.addEventListener("click", function () {
      if (comparisonPlaying) pauseComparison();
      else playComparison();
    });
  }

  if (comparisonRestart) {
    comparisonRestart.addEventListener("click", function () {
      var resume = comparisonPlaying;
      ensureComparisonLoaded().then(function () {
        pauseComparison();
        setComparisonTime(0);
        if (resume) playComparison();
      }).catch(function () {});
    });
  }

  if (comparisonScrubber) {
    comparisonScrubber.addEventListener("input", function () {
      var requestedProgress = Number(comparisonScrubber.value);
      ensureComparisonLoaded().then(function () {
        setComparisonTime((requestedProgress / 100) * comparisonDuration());
      }).catch(function () {});
    });
  }

  if (comparisonSpeed) {
    comparisonSpeed.addEventListener("change", function () {
      comparisonVideos.forEach(function (video) {
        video.playbackRate = Number(comparisonSpeed.value);
      });
    });
  }

  comparisonVideos.forEach(function (video) {
    video.addEventListener("loadedmetadata", renderComparisonProgress);
    video.addEventListener("error", function () {
      if (comparisonPlay) comparisonPlay.disabled = true;
    });
  });

  document.addEventListener("armbench:language", updateComparisonButton);
  updateComparisonButton();
  renderComparisonProgress();

  var ageSlider = document.querySelector("[data-age-slider]");
  var ageOutput = document.querySelector("[data-age-output]");
  var staleOutput = document.querySelector("[data-stale-output]");
  var windowOutput = document.querySelector("[data-window-output]");
  var alignedDescription = document.querySelector("[data-aligned-description]");
  var decisionState = document.querySelector("[data-decision-state]");
  var baselineTrack = document.querySelector("[data-baseline-track]");
  var alignedTrack = document.querySelector("[data-aligned-track]");
  var chunkSteps = 10;
  var replanSteps = 5;
  var controlPeriodMs = 50;
  var deadlineMs = 250;

  function makeActionCells(track) {
    if (!track || track.children.length) return;
    for (var index = 0; index < chunkSteps; index += 1) {
      var cell = document.createElement("span");
      cell.className = "action-cell";
      cell.textContent = "a" + index;
      cell.dataset.actionIndex = String(index);
      track.appendChild(cell);
    }
  }

  makeActionCells(baselineTrack);
  makeActionCells(alignedTrack);

  function renderTimingLab() {
    if (!ageSlider || !baselineTrack || !alignedTrack) return;
    var age = Number(ageSlider.value);
    var staleSteps = age <= 0 ? 0 : Math.ceil(age / controlPeriodMs);
    var selectedStop = staleSteps + replanSteps;
    var canExecute = age <= deadlineMs && selectedStop <= chunkSteps;

    if (ageOutput) ageOutput.textContent = age + " ms";
    if (staleOutput) {
      staleOutput.textContent = activeLanguage === "zh"
        ? staleSteps + " 个动作"
        : staleSteps + (staleSteps === 1 ? " action" : " actions");
    }

    Array.from(baselineTrack.children).forEach(function (cell, index) {
      if (index >= replanSteps) cell.dataset.state = "unused";
      else if (index < Math.min(staleSteps, replanSteps)) cell.dataset.state = "stale";
      else cell.dataset.state = "baseline-execute";
    });

    Array.from(alignedTrack.children).forEach(function (cell, index) {
      if (!canExecute) {
        cell.dataset.state = index < staleSteps ? "skipped" : "hold";
      } else if (index < staleSteps) {
        cell.dataset.state = "skipped";
      } else if (index < selectedStop) {
        cell.dataset.state = "execute";
      } else {
        cell.dataset.state = "unused";
      }
    });

    if (canExecute) {
      if (windowOutput) windowOutput.textContent = "a" + staleSteps + " → a" + (selectedStop - 1);
      if (alignedDescription) {
        alignedDescription.textContent = staleSteps === 0
          ? (activeLanguage === "zh" ? "没有过期前缀" : "no stale prefix")
          : (activeLanguage === "zh" ? "跳过 a0 到 a" + (staleSteps - 1) : "skip a0 to a" + (staleSteps - 1));
      }
      if (decisionState) {
        decisionState.dataset.state = "execute";
        decisionState.querySelector("strong").textContent = activeLanguage === "zh" ? "执行后缀" : "EXECUTE SUFFIX";
      }
    } else {
      if (windowOutput) windowOutput.textContent = activeLanguage === "zh" ? "无可执行窗口" : "no executable window";
      if (alignedDescription) alignedDescription.textContent = activeLanguage === "zh" ? "超出时限或剩余 horizon" : "deadline or horizon exceeded";
      if (decisionState) {
        decisionState.dataset.state = "hold";
        decisionState.querySelector("strong").textContent = activeLanguage === "zh" ? "保持 / 刷新" : "HOLD / REFRESH";
      }
    }

    ageSlider.setAttribute("aria-valuetext", activeLanguage === "zh" ? age + " 毫秒" : age + " milliseconds");
  }

  if (ageSlider) ageSlider.addEventListener("input", renderTimingLab);
  document.addEventListener("armbench:language", renderTimingLab);
  renderTimingLab();

  function fallbackCopy(text) {
    var input = document.createElement("textarea");
    input.value = text;
    input.setAttribute("readonly", "");
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.appendChild(input);
    input.select();
    var copied = document.execCommand("copy");
    input.remove();
    return copied;
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return fallbackCopy(text) ? Promise.resolve() : Promise.reject(new Error("Copy failed"));
  }

  document.querySelectorAll("[data-copy]").forEach(function (button) {
    var resetTimer = 0;
    button.setAttribute("aria-live", "polite");
    button.addEventListener("click", function () {
      window.clearTimeout(resetTimer);
      copyText(button.dataset.copy).then(function () {
        button.classList.add("copied");
        button.textContent = activeLanguage === "zh" ? "已复制" : "COPIED";
        resetTimer = window.setTimeout(function () {
          button.classList.remove("copied");
          button.textContent = activeLanguage === "zh" ? "复制" : "COPY";
        }, 1800);
      }).catch(function () {
        button.textContent = activeLanguage === "zh" ? "失败" : "FAILED";
      });
    });
    document.addEventListener("armbench:language", function () {
      if (!button.classList.contains("copied")) button.textContent = activeLanguage === "zh" ? "复制" : "COPY";
    });
    button.textContent = activeLanguage === "zh" ? "复制" : "COPY";
  });

  var revealItems = Array.from(document.querySelectorAll(".reveal"));
  if (reducedMotion.matches || !("IntersectionObserver" in window)) {
    revealItems.forEach(function (item) {
      item.classList.add("is-visible");
    });
  } else {
    var revealObserver = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8%", threshold: 0.12 });
    revealItems.forEach(function (item) {
      revealObserver.observe(item);
    });
  }

  var sectionLinks = Array.from(document.querySelectorAll(".nav-links a[href^='#']"));
  if ("IntersectionObserver" in window) {
    var sectionObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        sectionLinks.forEach(function (link) {
          var active = link.getAttribute("href") === "#" + entry.target.id;
          if (active) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-35% 0px -55%", threshold: 0 });
    sectionLinks.forEach(function (link) {
      var section = document.querySelector(link.getAttribute("href"));
      if (section) sectionObserver.observe(section);
    });
  }

  function handleReducedMotionChange(event) {
    if (event.matches) {
      heroUserPaused = true;
      pauseHero(false);
      pauseComparison();
      document.querySelectorAll(".reveal").forEach(function (item) {
        item.classList.add("is-visible");
      });
    }
  }

  if (typeof reducedMotion.addEventListener === "function") {
    reducedMotion.addEventListener("change", handleReducedMotionChange);
  } else {
    reducedMotion.addListener(handleReducedMotionChange);
  }

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      pauseHero(false);
      pauseComparison();
    } else if (!heroUserPaused && heroInView && !reducedMotion.matches && !saveData) {
      playHero(false);
    }
  });
})();
