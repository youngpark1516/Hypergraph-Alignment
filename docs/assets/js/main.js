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
document.addEventListener("DOMContentLoaded", async () => {
  const slot = document.getElementById("ablation-table-slot");
  if (!slot) return;

  try {
    const res = await fetch("assets/tables/ablation-table.html");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    slot.innerHTML = await res.text();
  } catch (err) {
    slot.innerHTML = '<p style="color:#b00020;">Failed to load ablation table.</p>';
    console.error("Ablation table load error:", err);
  }
});