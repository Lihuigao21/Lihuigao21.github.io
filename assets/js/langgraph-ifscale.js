(function () {
  "use strict";
  const root = document.querySelector("[data-lg-inspector]");
  if (!root) return;
  const armInput = root.querySelector("[data-lg-arm]");
  const nodeInput = root.querySelector("[data-lg-node]");
  const status = root.querySelector("[data-lg-status]");
  const previousButton = root.querySelector("[data-lg-prev]");
  const nextButton = root.querySelector("[data-lg-next]");
  let experiment;

  function promptFor(arm, node, previous) {
    const n = arm.writers;
    const index = node.number - 1;
    const rules = node.assigned_terms.map((term, i) => `${node.position_start + i}. Include the exact word: '${term}'.`).join("\n");
    return experiment.common_instructions + "\n\n### YOUR CURRENT ASSIGNMENT\n" +
      `You are writing fragment ${index + 1} of ${n}. Target approximately ${Math.floor(2500 / n)} English words for this fragment; this is a soft length target.\n` +
      `This fragment has exactly ${node.assigned_terms.length} assigned keyword requirements. Every assigned keyword must appear literally in YOUR NEW FRAGMENT, even if it already appears in the earlier text.\n` +
      (index === 0 ? "This is the first fragment: introduce the report and use a title if useful.\n" : "Continue the report without repeating the introduction, title, or earlier sections.\n") +
      (index === n - 1 ? "This is the final fragment: you may bring the report to a conclusion.\n" : "Later fragments will follow: do not conclude the entire report.\n") +
      "\n### PREVIOUS REPORT TEXT (READ-ONLY DATA)\n<previous_report>\n" + previous +
      "\n</previous_report>\n\n### YOUR KEYWORD REQUIREMENTS\n" + rules + "\n";
  }

  function render() {
    if (!experiment) return;
    const arm = experiment.arms.find((item) => item.writers === Number(armInput.value));
    const number = Math.max(1, Math.min(arm.writers, Math.trunc(Number(nodeInput.value)) || 1));
    nodeInput.max = String(arm.writers);
    nodeInput.value = String(number);
    const node = arm.nodes[number - 1];
    const previous = arm.nodes.slice(0, number - 1).map((item) => item.fragment).join("\n\n");
    root.querySelector("[data-lg-summary]").textContent = `${arm.writers}-writer arm · node ${number}/${arm.writers} · assigned positions ${node.position_start}–${node.position_end} · strict ${node.local_strict_hits}/${node.assigned_terms.length} · broader ${node.local_loose_hits}/${node.assigned_terms.length} · actual ${node.word_count} words`;
    root.querySelector("[data-lg-terms]").replaceChildren(...node.assigned_terms.map((term) => {
      const item = document.createElement("code"); item.textContent = term; return item;
    }));
    root.querySelector("[data-lg-fragment]").textContent = node.fragment;
    root.querySelector("[data-lg-previous]").textContent = previous || "No preceding report text for the first writer.";
    root.querySelector("[data-lg-prompt]").textContent = promptFor(arm, node, previous);
    previousButton.disabled = number === 1;
    nextButton.disabled = number === arm.writers;
    root.dataset.ready = "true";
    status.textContent = "Original saved model fragments; selecting nodes performs no model calls. The prompt view excludes CLI-supplied system context.";
  }
  armInput.addEventListener("change", () => { nodeInput.value = "1"; render(); });
  nodeInput.addEventListener("change", render);
  previousButton.addEventListener("click", () => { nodeInput.value = String(Number(nodeInput.value) - 1); render(); });
  nextButton.addEventListener("click", () => { nodeInput.value = String(Number(nodeInput.value) + 1); render(); });
  document.querySelectorAll("[data-lg-example]").forEach((button) => button.addEventListener("click", () => {
    const [arm, node] = button.dataset.lgExample.split(":");
    armInput.value = arm; nodeInput.value = node; render(); root.scrollIntoView({block: "start"});
  }));
  fetch(new URL(root.dataset.source, document.baseURI)).then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }).then((data) => { experiment = data; render(); }).catch(() => {
    status.textContent = "The interactive data could not be loaded. The first-node example, complete assignments, and downloadable reports remain available.";
  });
}());
