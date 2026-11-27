
"use strict";

const { ovar, general, cases } = window.videoTEDData;
const resultPanel = document.getElementById("result-panel");
const casePanel = document.getElementById("case-panel");
const resultButtons = [...document.querySelectorAll("[data-result]")];
const caseTabs = document.getElementById("case-tabs");

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function table(title, groups, measures, rows) {
  const section = element("section");
  const scroll = element("div", undefined, "table-scroll");
  scroll.tabIndex = 0;
  scroll.setAttribute("role", "region");
  scroll.setAttribute("aria-label", title);

  const grid = element("table", undefined, "result-table");
  grid.createCaption().textContent = title;
  const header = grid.createTHead();
  const heading = header.insertRow();
  const method = element("th", "Method");
  method.scope = "col";
  method.rowSpan = groups.length ? 2 : 1;
  heading.append(method);

  if (groups.length) {
    const subheading = header.insertRow();
    for (const group of groups) {
      const groupHeader = element("th", group);
      groupHeader.colSpan = measures.length;
      heading.append(groupHeader);
      for (const measure of measures) {
        const cell = element("th", measure);
        cell.scope = "col";
        subheading.append(cell);
      }
    }
  } else {
    for (const measure of measures) {
      const cell = element("th", measure);
      cell.scope = "col";
      heading.append(cell);
    }
  }

  const body = grid.createTBody();
  for (const [name, ...scores] of rows) {
    const row = body.insertRow();
    if (/ours|^Full/.test(name)) row.classList.add("ours");
    if (/^[∆Δ]/.test(name)) row.classList.add("gain");
    const label = element("th", name);
    label.scope = "row";
    row.append(label);
    for (const score of scores) row.append(element("td", score));
  }
  scroll.append(grid);
  section.append(scroll);
  return section;
}

function showResults(view) {
  for (const button of resultButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.result === view));
  }

  if (view === "ovar") {
    const overview = table(
      "Open-vocabulary action recognition (%) · Paper Table 1",
      ["K400", "HMDB51", "UCF101", "SSv2"], ["Base", "Novel", "HM"], ovar
    );
    overview.append(element("p", "HM is the harmonic mean of base and novel accuracy. † Locally evaluated under the paper’s unified protocol. Gains in percentage points were computed before rounding.", "table-note"));
    resultPanel.replaceChildren(overview);
    return;
  }

  if (view === "general") {
    const overview = table(
      "General video understanding (%) · Paper Table 2",
      ["MVBench", "Video-MME", "TempCompass", "SEED-Bench", "Mean"], ["Overall", "Action"], general
    );
    overview.append(element("p", "Averages are unweighted across the four benchmarks. † Locally evaluated using released checkpoints. Dashes indicate unavailable results.", "table-note"));
    resultPanel.replaceChildren(overview);
    return;
  }

  const measures = ["General avg.", "Action avg.", "SSv2 HM"];
  const variants = table("TCM and RTL (%) · Paper Table 3", [], measures, [
    ["Reference SFT only", "61.6", "66.4", "12.3"],
    ["GRPO only", "62.4", "68.5", "15.7"],
    ["GRPO + TCM", "62.8", "68.4", "17.2"],
    ["GRPO + RTL", "62.8", "69.2", "18.3"],
    ["Full VideoTED", "65.2", "71.7", "18.4"]
  ]);
  const components = table("Component ablations (%) · Paper Table 4", [], measures, [
    ["Base model", "60.8", "63.8", "13.2"],
    ["Without tree structure", "61.1", "64.5", "12.4"],
    ["Without tree context", "58.3", "63.6", "12.1"],
    ["Without EMA", "62.8", "66.5", "11.3"],
    ["Without reference analysis", "61.9", "65.3", "12.1"],
    ["Without reference weighting", "64.0", "68.6", "14.0"],
    ["Full VideoTED", "65.2", "71.7", "18.4"]
  ]);
  const hierarchy = table("Tree structure (%) · Supplement Table 6", [], measures, [
    ["Flat text", "61.1", "64.5", "12.4"],
    ["Shuffled hierarchy", "63.7", "70.1", "17.5"],
    ["Full VideoTED", "65.2", "71.7", "18.4"]
  ]);
  resultPanel.replaceChildren(variants, components, hierarchy);
}

function showCase(index) {
  const entry = cases[index];
  [...caseTabs.children].forEach((button, current) => {
    button.setAttribute("aria-pressed", String(index === current));
  });

  const frames = element("div", undefined, "case-frames");
  for (const [offset, file] of entry.frames.entries()) {
    const frameNumber = entry.indices[offset];
    const figure = element("figure");
    const picture = element("img");
    picture.src = "assets/cases/" + file;
    picture.alt = entry.title + ", sampled frame " + frameNumber;
    picture.loading = "lazy";
    figure.append(picture, element("figcaption", "Frame " + frameNumber));
    frames.append(figure);
  }

  const choices = element("ol", undefined, "case-choices");
  choices.type = "A";
  for (const [offset, choice] of entry.choices.entries()) {
    const correct = String.fromCharCode(65 + offset) === entry.truth;
    choices.append(element("li", correct ? choice + " (ground truth)" : choice, correct ? "correct" : ""));
  }

  const outputs = element("div", undefined, "two-col");
  for (const [name, output] of [["VideoTED", entry.ours], [entry.baseline, entry.comparison]]) {
    const details = element("details", undefined, "case-answer");
    const verdict = output.answer === entry.truth ? "correct" : "incorrect";
    details.dataset.verdict = verdict;
    details.append(
      element("summary", name + " · " + output.answer + " · " + verdict),
      element("p", output.reason)
    );
    outputs.append(details);
  }

  casePanel.replaceChildren(
    element("p", entry.task, "case-kicker"),
    element("h3", entry.question),
    frames, choices, outputs
  );
}

for (const button of resultButtons) {
  button.addEventListener("click", () => showResults(button.dataset.result));
}
cases.forEach((entry, index) => {
  const trigger = element("button", entry.title);
  trigger.type = "button";
  trigger.addEventListener("click", () => showCase(index));
  caseTabs.append(trigger);
});
showResults("ovar");
showCase(0);

const copyButton = document.getElementById("copy-citation");
copyButton.addEventListener("click", async () => {
  const bibtex = document.getElementById("bibtex");
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim());
    status.textContent = "Copied.";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Selected. Use Ctrl+C or ⌘C to copy.";
  }
});
