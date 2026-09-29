/* GLOD project page — shared behaviour for index.html and index-pt.html */
(function () {
  "use strict";

  const root = document.documentElement;
  root.classList.remove("no-js");
  const LANG = root.lang && root.lang.startsWith("pt") ? "pt" : "en";

  const T = {
    en: {
      all: "All families",
      copied: "Copied",
      identity: "flips = TV (no fitted constant)",
      kappaLine: "κ·√KL (κ fitted on this reference)",
      xTV: "per-token total variation",
      xKL: "√KL (per-token KL averaged, then rooted)",
      y: "flip rate",
      none: "No measurement of this family on this reference. Pick another family or reference.",
      selected: "selected",
    },
    pt: {
      all: "Todas as famílias",
      copied: "Copiado",
      identity: "flips = TV (sem constante ajustada)",
      kappaLine: "κ·√KL (κ ajustado nesta referência)",
      xTV: "variação total por token",
      xKL: "√KL (KL por token, média e depois raiz)",
      y: "taxa de flips",
      none: "Não há medição desta família nesta referência. Escolha outra família ou referência.",
      selected: "selecionada",
    },
  }[LANG];

  /* ---------- Theme ---------- */

  const THEME_KEY = "glod-theme";
  const themeBtn = document.getElementById("theme-toggle");
  function storedTheme() {
    try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; }
  }
  function applyTheme(t) {
    if (t) root.setAttribute("data-theme", t); else root.removeAttribute("data-theme");
  }
  function isDark() {
    const t = root.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  applyTheme(storedTheme());
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = isDark() ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* private mode */ }
      document.dispatchEvent(new Event("themechange"));
    });
  }
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if (!root.getAttribute("data-theme")) document.dispatchEvent(new Event("themechange"));
    });
  }

  /* ---------- Nav: shadow, active link, mobile menu ---------- */

  const nav = document.querySelector(".nav");
  const onScroll = () => nav && nav.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const menuBtn = document.getElementById("menu-toggle");
  if (menuBtn) {
    menuBtn.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll(".nav-links a").forEach((a) =>
      a.addEventListener("click", () => {
        nav.classList.remove("open");
        menuBtn.setAttribute("aria-expanded", "false");
      })
    );
  }

  const links = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
  const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  if ("IntersectionObserver" in window && targets.length) {
    const spy = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
        });
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    targets.forEach((t) => spy.observe(t));
  }

  /* ---------- Reveal on scroll + count-up ---------- */

  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const dec = parseInt(el.dataset.dec || "0", 10);
    const suffix = el.dataset.suffix || "";
    const fmt = (v) => v.toLocaleString(LANG === "pt" ? "pt-BR" : "en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
    if (reduced) { el.textContent = fmt(target); return; }
    const t0 = performance.now();
    const dur = 1100;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur);
      el.textContent = fmt(target * (1 - Math.pow(1 - k, 3)));
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  const revealEls = document.querySelectorAll(".reveal, [data-count]");
  if ("IntersectionObserver" in window && !reduced) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          e.target.classList.add("in");
          if (e.target.dataset.count) countUp(e.target);
          io.unobserve(e.target);
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => { el.classList.add("in"); if (el.dataset.count) countUp(el); });
  }

  /* ---------- Copy buttons ---------- */

  document.querySelectorAll(".copy").forEach((btn) => {
    btn.addEventListener("click", () => {
      const src = btn.dataset.target ? document.querySelector(btn.dataset.target) : btn.parentElement.querySelector("code, pre");
      if (!src || !navigator.clipboard) return;
      navigator.clipboard.writeText(src.textContent.trim()).then(() => {
        btn.classList.add("done");
        btn.setAttribute("aria-label", T.copied);
        setTimeout(() => btn.classList.remove("done"), 1500);
      });
    });
  });

  /* ---------- Data explorer ---------- */

  if (typeof GLOD_DATA === "undefined" || typeof Chart === "undefined") return;

  const FAMILIES = {
    rtn:       { en: "RTN", pt: "RTN", color: "#2563eb" },
    gptq:      { en: "GPTQ", pt: "GPTQ", color: "#0ea5e9" },
    awq:       { en: "AWQ", pt: "AWQ", color: "#06b6d4" },
    sparsegpt: { en: "SparseGPT", pt: "SparseGPT", color: "#7c3aed" },
    wanda:     { en: "Wanda", pt: "Wanda", color: "#a855f7" },
    magnitude: { en: "Magnitude", pt: "Magnitude", color: "#c026d3" },
    kv:        { en: "KV-cache quant.", pt: "Quant. do KV-cache", color: "#10b981" },
    skip:      { en: "Layer removal", pt: "Remoção de camadas", color: "#ef4444" },
    gauss:     { en: "Gaussian noise", pt: "Ruído gaussiano", color: "#64748b" },
    camada:    { en: "Single-layer RTN (control)", pt: "RTN em uma camada (controle)", color: "#f59e0b" },
    oficial:   { en: "Released checkpoints (control)", pt: "Checkpoints publicados (controle)", color: "#eab308" },
  };
  const CORPORA = { mix: "MIX (GSM8K + MMLU-PT)", gsm8k: "GSM8K", mmlu_en: "MMLU-en", wikitext: "WikiText (generated)", wikitext_nat: "WikiText (natural)" };
  if (LANG === "pt") { CORPORA.wikitext = "WikiText (gerado)"; CORPORA.wikitext_nat = "WikiText (natural)"; }
  const famName = (f) => (FAMILIES[f] ? FAMILIES[f][LANG] : f);
  const famColor = (f) => (FAMILIES[f] ? FAMILIES[f].color : "#999");

  const $ = (id) => document.getElementById(id);
  const corpusSel = $("x-corpus"), modelSel = $("x-model"), famSel = $("x-family"), klIn = $("x-kl");
  const kappaEl = $("x-kappa"), ratioEl = $("x-ratio"), readout = $("x-readout"), legendEl = $("x-legend");
  const viewBtns = document.querySelectorAll("[data-view]");
  let view = "tv";
  let selected = null;
  let chart = null;

  const css = (v) => getComputedStyle(root).getPropertyValue(v).trim();
  const num = (v, d) => v.toLocaleString(LANG === "pt" ? "pt-BR" : "en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (v) => num(v * 100, 2) + "%";
  const median = (a) => { const s = a.slice().sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

  function fillCorpora() {
    corpusSel.innerHTML = "";
    Object.keys(GLOD_DATA.measurements).forEach((c) => corpusSel.add(new Option(CORPORA[c] || c, c)));
    corpusSel.value = GLOD_DATA.measurements.mix ? "mix" : corpusSel.options[0].value;
  }
  function fillModels() {
    const prev = modelSel.value;
    modelSel.innerHTML = "";
    Object.keys(GLOD_DATA.measurements[corpusSel.value]).forEach((m) => modelSel.add(new Option(m, m)));
    const ms = GLOD_DATA.measurements[corpusSel.value];
    modelSel.value = ms[prev] ? prev : ms["Qwen3-4B"] ? "Qwen3-4B" : modelSel.options[0].value;
  }
  function points() {
    return (GLOD_DATA.measurements[corpusSel.value][modelSel.value] || []).filter((p) => p.flip > 0 && p.tv > 0 && p.kl > 0);
  }
  function kappa() { return GLOD_DATA.kappas[corpusSel.value + "|" + modelSel.value] || 0; }
  function fillFamilies() {
    const prev = famSel.value;
    famSel.innerHTML = "";
    famSel.add(new Option(T.all, "all"));
    Array.from(new Set(points().map((p) => p.family)))
      .sort((a, b) => Object.keys(FAMILIES).indexOf(a) - Object.keys(FAMILIES).indexOf(b))
      .forEach((f) => famSel.add(new Option(famName(f), f)));
    famSel.value = Array.from(famSel.options).some((o) => o.value === prev) ? prev : "all";
  }

  function refresh() {
    selected = null;
    readout.classList.remove("show");
    const pts = points();
    kappaEl.textContent = num(kappa(), 3);
    ratioEl.textContent = pts.length ? num(median(pts.map((p) => p.flip / p.tv)), 3) : "–";
    draw();
  }

  function draw() {
    const pts = points();
    const fam = famSel.value;
    const k = kappa();
    const xOf = (p) => (view === "tv" ? p.tv : Math.sqrt(p.kl));
    const byFam = {};
    pts.forEach((p) => {
      (byFam[p.family] = byFam[p.family] || []).push({ x: xOf(p), y: p.flip, raw: p });
    });

    const xs = pts.map(xOf), ys = pts.map((p) => p.flip);
    const lo = Math.min(...xs, ...ys) / 1.6, hi = Math.max(...xs, ...ys) * 1.6;
    const lineY = (x) => (view === "tv" ? x : k * x);
    const line = [];
    for (let i = 0; i <= 40; i++) { const x = lo * Math.pow(hi / lo, i / 40); line.push({ x, y: lineY(x) }); }

    const ink = css("--ink"), muted = css("--muted"), grid = css("--line"), accent = css("--accent");
    const datasets = [{
      type: "line", label: view === "tv" ? T.identity : T.kappaLine, data: line,
      borderColor: ink, borderWidth: 1.5, borderDash: view === "tv" ? [] : [5, 4], pointRadius: 0, order: 3,
    }];
    Object.keys(byFam).forEach((f) => {
      const dim = fam !== "all" && fam !== f;
      datasets.push({
        type: "scatter", label: famName(f), family: f, data: byFam[f],
        backgroundColor: famColor(f) + (dim ? "22" : "cc"), borderColor: dim ? "transparent" : famColor(f),
        borderWidth: 1, pointRadius: dim ? 3 : 4.5, pointHoverRadius: 7, order: dim ? 4 : 2,
      });
    });
    if (selected) {
      datasets.push({
        type: "scatter", label: T.selected, data: [{ x: xOf(selected), y: selected.flip, raw: selected }],
        backgroundColor: "transparent", borderColor: accent, borderWidth: 2.5, pointRadius: 11, pointHoverRadius: 11, order: 0,
      });
    }

    legendEl.innerHTML =
      `<span><i class="line" style="background:${ink}"></i>${view === "tv" ? T.identity : T.kappaLine}</span>` +
      Object.keys(byFam).map((f) => `<span><i style="background:${famColor(f)}"></i>${famName(f)}</span>`).join("");

    const axis = (title) => ({
      type: "logarithmic", min: lo, max: hi,
      title: { display: true, text: title, color: muted, font: { family: "Inter", size: 12 } },
      grid: { color: grid }, border: { color: grid },
      ticks: { color: muted, font: { family: "Inter", size: 12 }, maxTicksLimit: 6,
        callback: (v) => {
          const l = Math.log10(v);
          if (Math.abs(l - Math.round(l)) > 1e-6) return "";
          const sup = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
          return Math.round(l) === 0 ? "1" : "10" + String(Math.round(l)).split("").map((c) => sup[c]).join("");
        } },
    });

    const opts = {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: reduced ? 0 : 450, easing: "easeOutCubic" },
      interaction: { mode: "nearest", intersect: true },
      scales: { x: axis(view === "tv" ? T.xTV : T.xKL), y: axis(T.y) },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: css("--surface"), titleColor: ink, bodyColor: muted, borderColor: css("--line-strong"), borderWidth: 1,
          padding: 10, bodyFont: { family: "JetBrains Mono", size: 11 }, titleFont: { family: "Inter", size: 12, weight: "600" },
          filter: (item) => item.dataset.type !== "line",
          callbacks: {
            title: (items) => { const r = items[0].raw.raw; return r ? `${famName(r.family)} · ${r.config}` : ""; },
            label: (item) => { const r = item.raw.raw; return r ? [`KL ${num(r.kl, 4)} nat`, `TV ${pct(r.tv)}`, `flips ${pct(r.flip)}`] : ""; },
          },
        },
      },
      onClick: (evt, els) => {
        const hit = els.find((e) => chart.data.datasets[e.datasetIndex].type === "scatter");
        if (hit) select(chart.data.datasets[hit.datasetIndex].data[hit.index].raw);
      },
      onHover: (evt, els) => { evt.native.target.style.cursor = els.length ? "pointer" : "default"; },
    };

    if (chart) { chart.data.datasets = datasets; chart.options = opts; chart.update(); }
    else chart = new Chart($("x-chart"), { data: { datasets }, options: opts });
  }

  function select(p) {
    if (!p) return;
    selected = p;
    const k = kappa();
    const predKL = k * Math.sqrt(p.kl);
    const rel = (pred) => { const e = (pred - p.flip) / p.flip * 100; return (e > 0 ? "+" : "") + num(e, 1) + "%"; };
    $("x-cfg").textContent = `${famName(p.family)} · ${p.config}`;
    $("x-klv").textContent = num(p.kl, 4) + " nat";
    $("x-flip").textContent = pct(p.flip);
    $("x-tv").textContent = pct(p.tv);
    $("x-tv-err").textContent = rel(p.tv);
    $("x-kl-pred").textContent = pct(predKL);
    $("x-kl-err").textContent = rel(predKL);
    $("x-pr").textContent = num(p.flip / p.tv, 3);
    readout.classList.remove("show"); void readout.offsetWidth; readout.classList.add("show");
    draw();
  }

  function nearest() {
    const target = parseFloat(klIn.value);
    let pts = points();
    if (famSel.value !== "all") pts = pts.filter((p) => p.family === famSel.value);
    if (!pts.length || !(target > 0)) { $("x-cfg").textContent = T.none; readout.classList.add("show"); return; }
    const best = pts.reduce((a, b) => (Math.abs(Math.log(b.kl / target)) < Math.abs(Math.log(a.kl / target)) ? b : a));
    select(best);
  }

  corpusSel.addEventListener("change", () => { fillModels(); fillFamilies(); refresh(); });
  modelSel.addEventListener("change", () => { fillFamilies(); refresh(); });
  famSel.addEventListener("change", draw);
  $("x-go").addEventListener("click", nearest);
  klIn.addEventListener("keydown", (e) => { if (e.key === "Enter") nearest(); });
  viewBtns.forEach((b) => b.addEventListener("click", () => {
    view = b.dataset.view;
    viewBtns.forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    draw();
  }));
  document.addEventListener("themechange", draw);

  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  fillCorpora(); fillModels(); fillFamilies(); refresh();
})();
