(function () {
  "use strict";

  var root = document.documentElement;
  var languageButtons = Array.from(document.querySelectorAll("[data-set-lang]"));
  var savedLanguage = null;

  try {
    savedLanguage = window.localStorage.getItem("armbench-language");
  } catch (error) {
    savedLanguage = null;
  }

  var queryLanguage = new URLSearchParams(window.location.search).get("lang");
  var preferredLanguage = queryLanguage === "zh" || queryLanguage === "en"
    ? queryLanguage
    : savedLanguage === "zh" || savedLanguage === "en"
      ? savedLanguage
      : navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";

  function currentLanguage() {
    return root.dataset.lang === "en" ? "en" : "zh";
  }

  function setLanguage(language, persist) {
    var nextLanguage = language === "en" ? "en" : "zh";
    root.dataset.lang = nextLanguage;
    root.lang = nextLanguage === "zh" ? "zh-CN" : "en";
    document.title = nextLanguage === "zh"
      ? "ArmBench | VLA 运行时与评测"
      : "ArmBench | VLA Runtime & Evaluation";

    var description = document.querySelector('meta[name="description"]');
    if (description) {
      description.content = nextLanguage === "zh"
        ? "ArmBench：面向 action-chunk VLA 的时序对齐、运行时验证与可审计评测平台。"
        : "ArmBench: temporal alignment, runtime validation, and audit-ready evaluation for action-chunk VLA policies.";
    }

    languageButtons.forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.setLang === nextLanguage));
    });

    if (persist) {
      try {
        window.localStorage.setItem("armbench-language", nextLanguage);
      } catch (error) {
        // The language switch remains functional when storage is unavailable.
      }
    }

    updateTimeline();
  }

  languageButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      setLanguage(button.dataset.setLang, true);
    });
  });

  var menuButton = document.querySelector("[data-menu-button]");
  var navigation = document.querySelector(".nav-links");

  function setMenu(open) {
    if (!menuButton || !navigation) return;
    menuButton.setAttribute("aria-expanded", String(open));
    navigation.classList.toggle("open", open);
    document.body.classList.toggle("menu-open", open);
  }

  if (menuButton && navigation) {
    menuButton.addEventListener("click", function () {
      setMenu(menuButton.getAttribute("aria-expanded") !== "true");
    });

    navigation.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () { setMenu(false); });
    });

    window.addEventListener("keydown", function (event) {
      if (event.key === "Escape") setMenu(false);
    });

    window.addEventListener("resize", function () {
      if (window.innerWidth > 840) setMenu(false);
    });
  }

  var header = document.querySelector("[data-header]");
  function updateHeader() {
    if (header) header.classList.toggle("scrolled", window.scrollY > 24);
  }
  window.addEventListener("scroll", updateHeader, { passive: true });
  updateHeader();

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
    if (!track) return;
    for (var index = 0; index < chunkSteps; index += 1) {
      var cell = document.createElement("i");
      cell.className = "action-cell";
      cell.textContent = "a" + index;
      cell.dataset.actionIndex = String(index);
      track.appendChild(cell);
    }
  }

  makeActionCells(baselineTrack);
  makeActionCells(alignedTrack);

  function updateTimeline() {
    if (!ageSlider || !baselineTrack || !alignedTrack) return;

    var language = currentLanguage();
    var age = Number(ageSlider.value);
    var staleSteps = age <= 0 ? 0 : Math.ceil(age / controlPeriodMs);
    var selectedStop = staleSteps + replanSteps;
    var canExecute = age <= deadlineMs && selectedStop <= chunkSteps;
    var progress = Math.max(0, Math.min(100, age / Number(ageSlider.max) * 100));

    ageSlider.style.setProperty("--range-progress", progress + "%");
    if (ageOutput) ageOutput.textContent = age + " ms";
    if (staleOutput) {
      staleOutput.textContent = staleSteps + (language === "zh" ? " 个动作" : staleSteps === 1 ? " action" : " actions");
    }

    Array.from(baselineTrack.children).forEach(function (cell, index) {
      cell.className = "action-cell";
      if (index < replanSteps) cell.classList.add("selected");
      else cell.classList.add("unused");
      if (index < Math.min(staleSteps, replanSteps)) cell.classList.add("stale");
    });

    Array.from(alignedTrack.children).forEach(function (cell, index) {
      cell.className = "action-cell";
      if (!canExecute) {
        if (index < staleSteps) cell.classList.add("stale");
        else cell.classList.add("blocked");
      } else if (index < staleSteps) {
        cell.classList.add("stale");
      } else if (index < selectedStop) {
        cell.classList.add("selected");
      } else {
        cell.classList.add("unused");
      }
    });

    if (canExecute) {
      if (windowOutput) windowOutput.textContent = "a" + staleSteps + " → a" + (selectedStop - 1);
      if (alignedDescription) {
        alignedDescription.textContent = language === "zh"
          ? "跳过 a0–a" + Math.max(0, staleSteps - 1)
          : staleSteps === 0 ? "no stale prefix" : "skip a0–a" + (staleSteps - 1);
      }
      if (decisionState) {
        decisionState.classList.remove("hold");
        decisionState.querySelector("strong").textContent = language === "zh" ? "执行后缀" : "EXECUTE SUFFIX";
      }
    } else {
      if (windowOutput) windowOutput.textContent = language === "zh" ? "无可执行窗口" : "no executable window";
      if (alignedDescription) alignedDescription.textContent = language === "zh" ? "超出时限或 horizon" : "deadline or horizon exceeded";
      if (decisionState) {
        decisionState.classList.add("hold");
        decisionState.querySelector("strong").textContent = language === "zh" ? "保持 / 刷新" : "HOLD / REFRESH";
      }
    }
  }

  if (ageSlider) ageSlider.addEventListener("input", updateTimeline);

  document.querySelectorAll("[data-copy]").forEach(function (button) {
    button.addEventListener("click", function () {
      var value = button.dataset.copy;
      var copyPromise;

      if (navigator.clipboard && window.isSecureContext) {
        copyPromise = navigator.clipboard.writeText(value);
      } else {
        var input = document.createElement("textarea");
        input.value = value;
        input.setAttribute("readonly", "");
        input.style.position = "fixed";
        input.style.opacity = "0";
        document.body.appendChild(input);
        input.select();
        document.execCommand("copy");
        input.remove();
        copyPromise = Promise.resolve();
      }

      copyPromise.then(function () {
        button.classList.add("copied");
        button.setAttribute("aria-label", currentLanguage() === "zh" ? "已复制" : "Copied");
        window.setTimeout(function () {
          button.classList.remove("copied");
          button.setAttribute("aria-label", currentLanguage() === "zh" ? "复制命令" : "Copy command");
        }, 1800);
      });
    });
  });

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var autoplayVideo = document.querySelector("[data-autoplay-video]");
  if (reduceMotion.matches && autoplayVideo) {
    autoplayVideo.pause();
  } else if (autoplayVideo) {
    autoplayVideo.play().catch(function () {
      // The poster remains visible when a browser blocks muted playback.
    });
  }

  setLanguage(preferredLanguage, false);
})();
