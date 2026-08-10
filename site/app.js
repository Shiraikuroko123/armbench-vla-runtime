(() => {
  "use strict";

  const root = document.documentElement;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const saveData = Boolean(navigator.connection && navigator.connection.saveData);

  const getLanguage = () => {
    try {
      return localStorage.getItem("armbench-language") === "en" ? "en" : "zh";
    } catch {
      return "zh";
    }
  };

  const languageButtons = [...document.querySelectorAll("[data-set-lang]")];
  const updateLocalizedLabels = (language) => {
    document.querySelectorAll("[data-aria-play-zh]").forEach((control) => {
      const isPressed = control.getAttribute("aria-pressed") === "true";
      const key = isPressed ? `ariaPause${language === "zh" ? "Zh" : "En"}` : `ariaPlay${language === "zh" ? "Zh" : "En"}`;
      const label = control.dataset[key];
      if (label) control.setAttribute("aria-label", label);
    });

    const menuButton = document.querySelector("[data-menu-button]");
    if (menuButton) {
      const isOpen = menuButton.getAttribute("aria-expanded") === "true";
      const key = isOpen ? `ariaClose${language === "zh" ? "Zh" : "En"}` : `ariaOpen${language === "zh" ? "Zh" : "En"}`;
      const label = menuButton.dataset[key];
      if (label) menuButton.setAttribute("aria-label", label);
    }
  };

  const setLanguage = (language) => {
    root.dataset.lang = language;
    root.lang = language === "zh" ? "zh-CN" : "en";
    languageButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.setLang === language));
    });
    try {
      localStorage.setItem("armbench-language", language);
    } catch {
      // Language persistence is optional.
    }
    updateLocalizedLabels(language);
  };

  languageButtons.forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.setLang));
  });
  setLanguage(getLanguage());

  const menuButton = document.querySelector("[data-menu-button]");
  const navigation = document.querySelector("#site-nav");
  const closeMenu = () => {
    if (!menuButton || !navigation) return;
    navigation.classList.remove("is-open");
    menuButton.setAttribute("aria-expanded", "false");
    document.body.classList.remove("menu-open");
    updateLocalizedLabels(root.dataset.lang || "zh");
  };

  if (menuButton && navigation) {
    menuButton.addEventListener("click", () => {
      const open = menuButton.getAttribute("aria-expanded") !== "true";
      navigation.classList.toggle("is-open", open);
      menuButton.setAttribute("aria-expanded", String(open));
      document.body.classList.toggle("menu-open", open);
      updateLocalizedLabels(root.dataset.lang || "zh");
    });
    navigation.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
    window.addEventListener("resize", () => {
      if (window.innerWidth > 940) closeMenu();
    });
  }

  const loadVideo = (video) => {
    if (!video || video.dataset.loaded === "true") return;
    let attached = false;
    video.querySelectorAll("source[data-src]").forEach((source) => {
      source.src = source.dataset.src;
      attached = true;
    });
    if (attached) video.load();
    video.dataset.loaded = "true";
  };

  const lazyVideos = [...document.querySelectorAll("video[data-lazy-video]")];
  if ("IntersectionObserver" in window) {
    const videoObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting || entry.target.closest("[hidden]")) return;
        loadVideo(entry.target);
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "240px 0px" });
    lazyVideos.forEach((video) => videoObserver.observe(video));
  } else {
    lazyVideos.forEach(loadVideo);
  }

  const heroVideo = document.querySelector("[data-hero-video]");
  const heroToggle = document.querySelector("[data-video-toggle]");
  const syncHeroState = () => {
    if (!heroVideo || !heroToggle) return;
    heroToggle.setAttribute("aria-pressed", String(!heroVideo.paused));
    updateLocalizedLabels(root.dataset.lang || "zh");
  };

  if (heroVideo && heroToggle) {
    heroToggle.addEventListener("click", async () => {
      if (heroVideo.paused) {
        try {
          await heroVideo.play();
        } catch {
          // Native controls remain available if autoplay/play is blocked.
        }
      } else {
        heroVideo.pause();
      }
      syncHeroState();
    });
    heroVideo.addEventListener("play", syncHeroState);
    heroVideo.addEventListener("pause", syncHeroState);

    if (!reducedMotion.matches && !saveData) {
      heroVideo.play().catch(syncHeroState);
    } else {
      heroVideo.pause();
      syncHeroState();
    }
  }

  reducedMotion.addEventListener("change", (event) => {
    if (event.matches && heroVideo) heroVideo.pause();
  });

  document.querySelectorAll("[data-tabs]").forEach((tabGroup) => {
    const tabs = [...tabGroup.querySelectorAll("[role='tab']")];
    const panels = [...tabGroup.querySelectorAll("[role='tabpanel']")];

    const activateTab = (nextTab) => {
      tabs.forEach((tab) => {
        const active = tab === nextTab;
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
      });
      panels.forEach((panel) => {
        const active = panel.id === nextTab.dataset.tab;
        panel.hidden = !active;
        if (active) {
          panel.querySelectorAll("video[data-lazy-video]").forEach(loadVideo);
        } else {
          panel.querySelectorAll("video").forEach((video) => video.pause());
        }
      });
    };

    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => activateTab(tab));
      tab.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        let nextIndex = index;
        if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
        if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = tabs.length - 1;
        tabs[nextIndex].focus();
        activateTab(tabs[nextIndex]);
      });
    });
  });

  document.querySelectorAll("[data-copy-code]").forEach((button) => {
    button.addEventListener("click", async () => {
      const code = button.parentElement && button.parentElement.querySelector("code");
      if (!code) return;
      const language = root.dataset.lang || "zh";
      const original = language === "zh" ? "复制" : "Copy";
      try {
        await navigator.clipboard.writeText(code.textContent.trim());
        button.textContent = language === "zh" ? "已复制" : "Copied";
      } catch {
        button.textContent = language === "zh" ? "复制失败" : "Copy failed";
      }
      window.setTimeout(() => {
        button.textContent = original;
      }, 1600);
    });
  });

  const sections = [...document.querySelectorAll("main section[id]")];
  const navLinks = [...document.querySelectorAll("#site-nav a[href^='#']")];
  if ("IntersectionObserver" in window) {
    const sectionObserver = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      navLinks.forEach((link) => {
        const current = link.getAttribute("href") === `#${visible.target.id}`;
        if (current) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
    }, { rootMargin: "-20% 0px -68%", threshold: [0, 0.15, 0.4] });
    sections.forEach((section) => sectionObserver.observe(section));
  }
})();
