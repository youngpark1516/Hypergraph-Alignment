// smooth anchor scrolling
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener("click", (e) => {
    const id = a.getAttribute("href");
    const el = document.querySelector(id);
    if (!el) return;
    e.preventDefault();
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

// footer year
document.getElementById("year").textContent = new Date().getFullYear();
async function loadTableSlot(slotId, url, label) {
  const slot = document.getElementById(slotId);
  if (!slot) return;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    slot.innerHTML = await res.text();
  } catch (err) {
    slot.innerHTML = `<p style="color:#b00020;">Failed to load ${label}.</p>`;
    console.error(`${label} load error:`, err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadTableSlot("ablation-table-slot", "assets/tables/ablation-table.html", "ablation table");
  loadTableSlot("l1-table-slot",       "assets/tables/l1-improvement-table.html", "L1 improvement table");
});