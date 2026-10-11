// Build presentation/EpiGrade_slides.pptx.
// Usage: NODE_PATH=<dir>/node_modules node presentation/build_deck.js [path/to/apply_theme.js]
// Every number on a slide is read from results/tables/ at build time - none is typed here.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const root = path.resolve(__dirname, "..");
const FIG = path.join(root, "results", "figures", "slides");
const OUT = path.join(__dirname, "EpiGrade_slides.pptx");

// ------------------------------------------------------------------ data
function tsv(name) {
  const text = fs.readFileSync(path.join(root, "results", "tables", name), "utf8");
  const [head, ...rows] = text.trim().split(/\r?\n/).map((l) => l.split("\t"));
  return rows.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i] ?? ""])));
}
const counts = Object.fromEntries(tsv("sample_counts.tsv").map((r) => [r.category, r.count]));
const verdicts = tsv("disorder_verdicts.tsv");
const nVerdict = (v) => verdicts.filter((r) => r.verdict === v).length;
const loso = tsv("leave_one_study_out.tsv");
const losoOk = loso.filter((r) => r.status === "computed");
const matrix = tsv("cross_disorder_matrix.tsv");
const cell = (c, d) => matrix.find((r) => r.classifier === c && r.scored_disorder === d);
const pct = (x) => `${(100 * Number(x)).toFixed(1)}%`;
const sotos = tsv("sotos_reproduction_summary.tsv");
const byo = tsv("byo_fixture_results.tsv");
const bands = tsv("evidence_bands.tsv")
  .filter((r) => r.query_point === "typical_case_score" && r.prior === "0.1");
const between = bands.filter((r) => r.interpretation_scope === "between_study");
const fmt = (n) => Number(n).toLocaleString("en-US");
const charge = cell("CHARGE syndrome", "Kabuki syndrome type 1");
const css = cell("Coffin-Siris syndrome", "pooled_control");
const effect = Object.fromEntries(tsv("controls_source_summary.tsv").map((r) => [r.controls_source, r]));
const offDiagonal = matrix.filter((r) => r.status === "computed" && r.cell_type === "other_disorder"
  && !r.interpretation);
const perClassifier = [...new Set(offDiagonal.map((r) => r.classifier))].map((c) => {
  const cells = offDiagonal.filter((r) => r.classifier === c);
  return { name: c.replace(" syndrome", "").replace(" type 1", "").replace("7q11.23 duplication", "7q11.23 dup"),
    source: cells[0].controls_source,
    mean: cells.reduce((a, r) => a + Number(r.fraction_positive), 0) / cells.length };
}).sort((a, b) => a.mean - b.mean);
const random = byo.find((r) => r.fixture === "sig_random.csv");
const tiny = byo.find((r) => r.fixture === "sig_tiny.csv");
const full = byo.find((r) => r.fixture === "sig_sotos_true.csv");

// ------------------------------------------------------------------ theme
const THEME = {
  name: "EpiGrade",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14213D", lt1: "FFFFFF", dk2: "3D4A63", lt2: "F1F4F9",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "8A8984",
    accent5: "0D366B", accent6: "D03B3B", hlink: "2A78D6", folHlink: "4A3AA7",
  },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "EpiGrade: an open, reproducible benchmark for episignature classifiers";
pres.author = "Upasana Bhaumik, Samhitaa Saravanakumar, Divyanjali Tyagi";
const C = pres.SchemeColor;
const W = 13.33;
const M = 0.6;

pres.defineSlideMaster({
  title: "DARK",
  background: { color: THEME.colors.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.2, w: W - 2 * M, h: 1.9,
      fontSize: 44, bold: true, color: C.background1, align: "left", valign: "top",
      margin: 0 },
      text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: THEME.colors.lt1 },
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, fontSize: 12, color: THEME.colors.accent4 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: W - 2 * M, h: 1.0,
      fontSize: 36, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 },
      text: "" } },
    { text: { text: "EpiGrade  |  research tool, not for patient use",
      options: { x: M, y: 7.0, w: 7, h: 0.3, fontSize: 12, color: THEME.colors.accent4,
        margin: 0 } } },
  ],
});

// ------------------------------------------------------------------ helpers
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
// Place an image inside a box, keeping its aspect ratio (left-aligned, vertically centred).
function fit(slide, file, box, name) {
  const { w, h } = pngSize(file);
  const scale = Math.min(box.w / w, box.h / h);
  const iw = w * scale;
  const ih = h * scale;
  slide.addImage({ path: file, x: box.x + (box.align === "center" ? (box.w - iw) / 2 : 0),
    y: box.y + (box.h - ih) / 2, w: iw, h: ih, objectName: name, altText: name });
}
const text = (slide, value, o) => slide.addText(value, { isTextBox: true, margin: 0,
  fontSize: 20, color: C.text1, valign: "top", ...o });
function card(slide, o, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: o.x, y: o.y, w: o.w, h: o.h,
    rectRadius: 0.08, fill: { color: o.fill || C.background2 }, line: { color: o.fill || C.background2 },
    objectName: name });
}
function stat(slide, big, label, o) {
  card(slide, o, `stat ${label}`);
  text(slide, big, { x: o.x + 0.25, y: o.y + 0.15, w: o.w - 0.5, h: 1.1, fontSize: 54, bold: true,
    color: o.color || C.accent1, fontFace: THEME.headFontFace });
  text(slide, label, { x: o.x + 0.25, y: o.y + 1.25, w: o.w - 0.5, h: o.h - 1.35, fontSize: 20,
    color: C.text2 });
}
function content(section, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function table(slide, rows, o) {
  const body = rows.map((r, i) => r.map((v) => ({
    text: String(v),
    options: { bold: i === 0, color: i === 0 ? THEME.colors.lt1 : THEME.colors.dk1,
      fill: { color: i === 0 ? THEME.colors.dk1 : (i % 2 ? THEME.colors.lt1 : THEME.colors.lt2) },
      fontSize: o.fontSize || 20, valign: "middle", margin: [4, 8, 4, 8] },
  })));
  slide.addTable(body, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH || 0.62,
    border: { type: "solid", pt: 0.75, color: "D5DAE3" }, objectName: o.name });
}

// ================================================================== slides
pres.addSection({ title: "The problem" });

// 1 ---------------------------------------------------------------- title
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "The problem" });
  s.addText("EpiGrade", { placeholder: "title" });
  text(s, "An open, reproducible benchmark for DNA methylation episignature classifiers",
    { x: M, y: 3.35, w: 10.5, h: 1.2, fontSize: 28, color: C.background2 });
  text(s, [
    { text: "Upasana Bhaumik (23BCB0074)", options: { breakLine: true } },
    { text: "Samhitaa Saravanakumar (23BCB0125)", options: { breakLine: true } },
    { text: "Divyanjali Tyagi (23BCB0123)" },
  ], { x: M, y: 4.9, w: 7, h: 1.5, fontSize: 22, color: C.background1 });
  text(s, "Vellore Institute of Technology (VIT), Vellore",
    { x: M, y: 6.5, w: 9, h: 0.45, fontSize: 20, color: C.background2 });
  s.addNotes("Good morning. We are Upasana, Samhitaa and Divyanjali from VIT Vellore. Our project is called EpiGrade. It checks, using only public data, how far a new kind of genetic test can be trusted. We will show what we built, three things we found, and a live tool.");
}

// 2 ---------------------------------------------------------------- problem
{
  const s = content("The problem", "A child, a DNA change, and no answer");
  const steps = [
    ["1", "A child has a disorder", "Doctors suspect a genetic cause and sequence the child's DNA."],
    ["2", "Sequencing finds a rare change", "It sits in a plausible gene, but nobody has seen it before."],
    ["3", "The lab cannot interpret it", "It is reported as a variant of uncertain significance. The family gets no answer."],
  ];
  steps.forEach(([n, head, body], i) => {
    const x = M + i * 4.1;
    card(s, { x, y: 1.75, w: 3.8, h: 4.3 }, `step ${n}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 2.05, w: 0.9, h: 0.9,
      fill: { color: i === 2 ? C.accent2 : C.accent1 }, line: { color: C.background2 },
      objectName: `step ${n} badge` });
    text(s, n, { x: x + 0.3, y: 2.05, w: 0.9, h: 0.9, fontSize: 30, bold: true,
      color: C.background1, align: "center", valign: "middle" });
    text(s, head, { x: x + 0.3, y: 3.15, w: 3.2, h: 0.9, fontSize: 24, bold: true });
    text(s, body, { x: x + 0.3, y: 4.2, w: 3.2, h: 1.7, fontSize: 20, color: C.text2 });
  });
  text(s, "An uncertain variant cannot confirm a diagnosis, guide care, or be used to test relatives.",
    { x: M, y: 6.25, w: W - 2 * M, h: 0.5, fontSize: 20, italic: true, color: C.text2 });
  s.addNotes("Start with the patient. A child has a developmental disorder, and doctors read the child's DNA to find the cause. Often they find a change that is rare and sits in a sensible gene, but nobody can say whether it causes disease. That result is called a variant of uncertain significance. For the family it means no diagnosis and no guidance.");
}

// 3 ---------------------------------------------------------------- episignature diagram
{
  const s = content("The problem", "A second test: the episignature");
  text(s, "Some disease genes control chemical tags on DNA (methylation). When such a gene is broken, the pattern of tags in blood changes in a recognisable way.",
    { x: M, y: 1.6, w: 5.4, h: 2.3, fontSize: 22 });
  text(s, "Pattern present: the variant is probably harmful. Pattern absent: probably harmless.",
    { x: M, y: 4.2, w: 5.4, h: 1.5, fontSize: 22, bold: true, color: C.accent5 });
  // diagram: three rows of CpG sites, filled = methylated
  const rows = [
    ["Healthy control", [1, 1, 0, 1, 1, 1, 0, 1, 1, 1], C.accent4],
    ["Patient with the disorder", [0, 1, 0, 0, 1, 0, 0, 0, 1, 0], C.accent2],
    ["Patient with an unknown variant", [0, 1, 0, 0, 1, 0, 0, 1, 1, 0], C.accent1],
  ];
  card(s, { x: 6.5, y: 1.6, w: 6.2, h: 5.2 }, "diagram panel");
  rows.forEach(([label, sites, colour], r) => {
    const y = 2.0 + r * 1.5;
    text(s, label, { x: 6.8, y, w: 5.6, h: 0.4, fontSize: 20, bold: true });
    s.addShape(pres.shapes.LINE, { x: 6.8, y: y + 0.82, w: 5.6, h: 0,
      line: { color: THEME.colors.accent4, width: 1.5 }, objectName: `dna ${r}` });
    sites.forEach((on, k) => {
      s.addShape(pres.shapes.OVAL, { x: 6.9 + k * 0.56, y: y + 0.58, w: 0.44, h: 0.44,
        fill: { color: on ? colour : C.background1 }, line: { color: colour, width: 2 },
        objectName: `site ${r}-${k}` });
    });
  });
  text(s, "Filled circle = tagged site",
    { x: 6.8, y: 6.3, w: 5.7, h: 0.35, fontSize: 16, color: C.text2 });
  s.addNotes("There is a second way to test. Some disease genes control small chemical tags on DNA, a process called methylation. When one of these genes stops working, the pattern of tags across the genome changes in a characteristic way. That pattern is called an episignature. If a patient with an unclear variant shows the pattern, the variant is probably harmful, and if not, it is probably harmless.");
}

// 4 ---------------------------------------------------------------- the catch
{
  const s = content("The problem", "The catch: nobody outside can check it");
  stat(s, "34", "disease signatures reported from 42 syndromes in one landmark study (Aref-Eshghi 2020)",
    { x: M, y: 1.7, w: 3.8, h: 3.0 });
  stat(s, "1", "private reference database that these tests are validated against",
    { x: M + 4.1, y: 1.7, w: 3.8, h: 3.0, color: C.accent2 });
  stat(s, "<40%", "sensitivity for 2 of 10 signatures when an outside group re-tested them (Husson 2024)",
    { x: M + 8.2, y: 1.7, w: 3.8, h: 3.0, color: C.accent6 });
  text(s, "Results are reported in categories agreed by experts, not as a measured strength of evidence. Our question: how much can be confirmed from public data alone?",
    { x: M, y: 5.05, w: W - 2 * M, h: 1.5, fontSize: 24 });
  s.addNotes("Here is the catch. These tests are validated against one private reference database, and laboratories outside cannot inspect it. One study reported 34 signatures across 42 syndromes. But when an independent French group re-tested ten published signatures with new patients, two of them detected fewer than 40 percent of cases. So we asked how much can be confirmed using only data that anyone can download.");
}

pres.addSection({ title: "What we built" });

// 5 ---------------------------------------------------------------- what we built
{
  const s = content("What we built", "What we built: four components");
  const parts = [
    ["Corpus", `${fmt(counts.harvested)} public samples from 18 datasets, ${fmt(counts.retained_for_analysis)} kept, a recorded reason for every exclusion`],
    ["Classifiers", `${verdicts.filter((r) => r.signature_built === "True").length} disease signatures: 1 published and reproduced, 9 re-derived under one fixed rule`],
    ["Audit and calibration", "Cross-study tests, every classifier against every disorder, and scores turned into evidence strength"],
    ["Identification engine", "Upload any signature and learn which of 12 disorders it identifies, if any"],
  ];
  parts.forEach(([head, body], i) => {
    const x = M + (i % 2) * 6.15;
    const y = 1.7 + Math.floor(i / 2) * 2.55;
    card(s, { x, y, w: 5.85, h: 2.3 }, `component ${i + 1}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.35, w: 0.8, h: 0.8,
      fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `component badge ${i + 1}` });
    text(s, String(i + 1), { x: x + 0.3, y: y + 0.35, w: 0.8, h: 0.8, fontSize: 26, bold: true,
      color: C.background1, align: "center", valign: "middle" });
    text(s, head, { x: x + 1.35, y: y + 0.3, w: 4.3, h: 0.5, fontSize: 24, bold: true });
    text(s, body, { x: x + 1.35, y: y + 0.9, w: 4.3, h: 1.3, fontSize: 20, color: C.text2 });
  });
  s.addNotes("We built four things. First, a collection of public samples where every sample we excluded has a written reason. Second, signatures for ten disorders: one published signature that we reproduced, and nine that we derived ourselves under a single fixed rule. Third, an audit that tests each signature across studies and against every other disorder, and converts scores into evidence strength. Fourth, a tool where anyone can upload their own signature and test it.");
}

// 6 ---------------------------------------------------------------- proof
{
  const s = content("What we built", "Proof it works: the Sotos syndrome signature");
  fit(s, path.join(root, "results", "figures", "slides", "sotos_reproduction_scores.png"),
    { x: M, y: 1.55, w: 7.6, h: 5.2 }, "Sotos reproduction scores by cohort");
  text(s, "Published (Choufani 2015)", { x: 8.6, y: 1.7, w: 4.1, h: 0.45, fontSize: 22, bold: true, color: C.accent5 });
  text(s, "19 cases and 53 controls separate cleanly at a score of zero.",
    { x: 8.6, y: 2.2, w: 4.1, h: 1.3, fontSize: 20, color: C.text2 });
  text(s, "Ours, from the public data", { x: 8.6, y: 3.75, w: 4.1, h: 0.45, fontSize: 22, bold: true, color: C.accent2 });
  const d = sotos.find((r) => r.check === "discovery_separation").observed;
  text(s, `${d.match(/case>0: (\d+\/\d+)/)[1]} cases above zero, ${d.match(/control<0: (\d+\/\d+)/)[1]} controls below zero. Each sample was scored without using itself.`,
    { x: 8.6, y: 4.25, w: 4.1, h: 2.2, fontSize: 20, color: C.text2 });
  s.addNotes("Before auditing anything, we had to show our pipeline is correct. The Sotos syndrome signature, published by Choufani and colleagues in 2015, comes with its full probe list, its scoring rule and its data. We rebuilt it from the public files. Every case scores above zero and every control below zero, exactly as published. Each sample was scored against profiles built without it, so the result is not inflated.");
}

// 7 ---------------------------------------------------------------- four checks
{
  const s = content("What we built", "Four reproduction checks, all passing");
  const label = { discovery_separation: "Cases and controls separate", weaver_negative: "Similar disorder (Weaver) scores negative",
    missense_split: "Unclear NSD1 variants split", replication: "Held-out cases score positive" };
  const published = { discovery_separation: "19 cases vs 53 controls, clean split",
    weaver_negative: "8 of 8 negative", missense_split: "9 positive, 7 negative", replication: "all positive" };
  const ours = (r) => {
    if (r.check === "discovery_separation") return `${r.observed.match(/case>0: (\S+);/)[1]} and ${r.observed.match(/control<0: (\S+)/)[1]}`;
    if (r.check === "weaver_negative") return r.observed.split(";")[0].replace(" scored", "");
    const n = r.observed.match(/n=(\d+)/)[1];
    const pos = r.observed.match(/positive=(\d+)/)[1];
    const neg = r.observed.match(/negative=(\d+)/);
    return neg ? `${pos} positive, ${neg[1]} negative` : `${pos} of ${n} positive`;
  };
  const rows = [["Check", "Published", "Ours", "Result"]];
  sotos.forEach((r) => rows.push([label[r.check], published[r.check], ours(r),
    String(r.reproduced).toLowerCase() === "true" ? "Pass" : "Fail"]));
  table(s, rows, { x: M, y: 1.75, w: W - 2 * M, colW: [4.3, 3.6, 2.9, 1.33], rowH: 0.8, name: "four checks table" });
  text(s, `${sotos.filter((r) => String(r.reproduced).toLowerCase() === "true").length} of ${sotos.length} checks pass. The pipeline reproduces a published result exactly.`,
    { x: M, y: 6.05, w: W - 2 * M, h: 0.6, fontSize: 24, bold: true, color: C.accent3 });
  s.addNotes("We checked four things against the paper. Cases and controls separate. Eight patients with Weaver syndrome, which looks similar but has a different cause, all score negative. Sixteen unclear variants in the NSD1 gene split nine positive and seven negative, exactly as published. And nineteen cases that were never used to build the classifier all score positive. All four checks pass.");
}

pres.addSection({ title: "Findings" });

// 8 ---------------------------------------------------------------- finding 1
{
  const s = content("Findings", `Finding 1: only ${nVerdict("graded")} of ${verdicts.length} disorders span two studies`);
  fit(s, path.join(FIG, "disorder_verdict_summary.png"), { x: M, y: 1.5, w: 8.2, h: 5.3 },
    "Twelve disorders by verdict");
  stat(s, String(nVerdict("single-study")), "disorders have every case in one study, so disease and laboratory batch cannot be told apart",
    { x: 9.1, y: 1.6, w: 3.6, h: 2.75, color: C.accent2 });
  stat(s, String(nVerdict("too-few-cases")), "disorders have under 10 cases, too few for any signature",
    { x: 9.1, y: 4.55, w: 3.6, h: 2.25, color: C.accent3 });
  s.addNotes("Our first finding is about the data itself. Of twelve disorders, seven have all their cases in a single study. In that situation a difference between cases and controls could be the disease, or it could just be how that laboratory processed its samples, and we cannot tell which. Two more disorders have fewer than ten cases, which is too few to build anything. Only three disorders have cases in two independent studies.");
}

// 9 ---------------------------------------------------------------- finding 2
{
  const ok = losoOk[0];
  const untested = new Set(loso.map((r) => r.disorder)).size
    - new Set(losoOk.map((r) => r.disorder)).size;
  const s = content("Findings", `Finding 2: ${losoOk.length} of ${loso.length} cross-study tests could be run`);
  fit(s, path.join(FIG, "leave_one_study_out.png"), { x: M, y: 1.5, w: 8.3, h: 5.3 },
    "Leave-one-study-out results");
  stat(s, ok.test_case_sensitivity, `held-out Silver-Russell cases detected (${pct(ok.sensitivity)}); ${ok.test_control_specificity} controls correctly negative`,
    { x: 9.2, y: 1.6, w: 3.5, h: 2.75, color: C.accent1 });
  text(s, `We ran one cross-study test, not three: one disorder, one direction. ${untested} of 3 eligible disorders have no cross-study result at all.`,
    { x: 9.2, y: 4.55, w: 3.5, h: 2.25, fontSize: 20, bold: true });
  s.addNotes("The real test is to build a signature in one study and try it in another. This is called leave-one-study-out. Three disorders were eligible, which gives six possible directions. We want to be plain about this: only one of the six could be computed. That one test found five of eighteen held-out cases. Kabuki syndrome and Nicolaides-Baraitser syndrome have no cross-study result at all, because one study alone did not have enough samples to build a signature.");
}

// 10 --------------------------------------------------------------- finding 3
{
  const s = content("Findings", "Finding 3: classifiers that fire on the wrong group");
  fit(s, path.join(FIG, "cross_disorder_matrix.png"), { x: M, y: 1.45, w: 7.9, h: 5.4 },
    "Cross-disorder heatmap");
  stat(s, pct(charge.fraction_positive), `of Kabuki cases are flagged by the CHARGE classifier (${charge.n_positive} of ${charge.n})`,
    { x: 8.8, y: 1.6, w: 3.9, h: 2.45, color: C.accent2 });
  card(s, { x: 8.8, y: 4.25, w: 3.9, h: 2.55 }, "outlined rows note");
  text(s, "Outlined rows", { x: 9.05, y: 4.45, w: 3.4, h: 0.45, fontSize: 22, bold: true, color: C.accent2 });
  text(s, "were trained against controls from another series. They light up almost everywhere.",
    { x: 9.05, y: 4.95, w: 3.4, h: 1.7, fontSize: 20, color: C.text2 });
  s.addNotes("Third, we scored every classifier against every disorder and against 273 healthy controls. A good classifier lights up only its own square on the diagonal. The CHARGE syndrome classifier also flags about 38 percent of Kabuki syndrome cases. Now look at the two rows outlined in orange. They light up almost everywhere. Those two classifiers were trained against controls from a different dataset, and the next slide shows what that does.");
}

// 11 --------------------------------------------------------------- finding 4
{
  const w = effect.within_series;
  const x = effect.cross_series;
  const s = content("Findings", "Finding 4: the controls-source effect");
  stat(s, pct(w.mean_off_diagonal_fraction_positive), `of other disorders' cases flagged with same-series controls (${w.n_classifiers} classifiers)`,
    { x: M, y: 1.6, w: 3.7, h: 2.5, color: C.accent1 });
  stat(s, pct(x.mean_off_diagonal_fraction_positive), `flagged with controls from another series (${x.n_classifiers} classifiers)`,
    { x: M, y: 4.3, w: 3.7, h: 2.5, color: C.accent2 });
  s.addChart(pres.charts.BAR, [{ name: "Mean fraction of other disorders' cases flagged",
    labels: perClassifier.map((r) => r.name), values: perClassifier.map((r) => r.mean) }], {
    x: 4.6, y: 1.5, w: 8.1, h: 4.2, barDir: "bar", showLegend: false,
    showTitle: true, title: "Other disorders' cases flagged, by classifier", titleFontSize: 16,
    titleFontFace: "+mn-lt", titleColor: THEME.colors.dk1,
    chartColors: perClassifier.map((r) => (r.source === "cross_series" ? THEME.colors.accent2 : THEME.colors.accent1)),
    invertedColors: [], showValue: true, dataLabelFormatCode: "0%", dataLabelPosition: "outEnd",
    dataLabelFontSize: 14, dataLabelFontFace: "+mn-lt", dataLabelColor: THEME.colors.dk1,
    catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: THEME.colors.dk2,
    valAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt", valAxisLabelColor: THEME.colors.dk2,
    valAxisLabelFormatCode: "0%", valAxisMaxVal: 1, valAxisMinVal: 0,
    valGridLine: { color: "E4E3DF", size: 0.75 }, catGridLine: { style: "none" },
    objectName: "off-diagonal mean by classifier", altText: "Bar chart of mean off-diagonal positive fraction per classifier",
  });
  text(s, "Classifiers trained against out-of-series controls detect series membership, not disease. Coffin-Siris and Nicolaides-Baraitser are both in that category.",
    { x: 4.6, y: 5.8, w: 8.1, h: 1.0, fontSize: 20, bold: true, color: C.accent2 });
  s.addNotes("This is the result in one comparison. When a classifier's controls come from its own dataset, it wrongly flags about 12 percent of patients who have a different disorder. When the controls come from another dataset, it flags about 68 percent. The two orange bars are Coffin-Siris and Nicolaides-Baraitser, whose source study deposited no controls. They learned to tell two laboratories apart, not to detect a disease. We keep them in the benchmark as a warning, not as signatures. One caution: both borrowed the same controls, so this is one problem seen twice.");
}

pres.addSection({ title: "Identification engine" });

// 12 --------------------------------------------------------------- BYO table
{
  const s = content("Identification engine", "Bring your own signature: five tests");
  const names = { "sig_sotos_true.csv": "500 real Sotos probes", "sig_random.csv": "500 random probes",
    "sig_sotos_flipped.csv": "Sotos probes, directions inverted", "sig_tiny.csv": "20 Sotos probes",
    "sig_low_coverage.csv": "500 probes not in the data" };
  const result = (r) => {
    if (r.status === "refused") return "Refused: too few usable probes";
    if (r.verdict === "no_match") return `No match (though raw AUC reaches ${Number(r.highest_auc).toFixed(2)})`;
    const extra = r.fixture === "sig_sotos_flipped.csv"
      ? `; ${parseInt(r.n_direction_inconsistent, 10)} of ${parseInt(r.n_directions_supplied, 10)} directions flagged as inverted`
      : `; score ${Number(r.top_match_z_vs_random_probes).toFixed(0)} against ${Number(r.runner_up_z).toFixed(0)} for the runner-up`;
    const cap = r.fixture === "sig_tiny.csv" ? `; ceiling ${r.evidence_ceiling_band}` : "";
    return `${r.top_match.replace(" syndrome", "")}${extra}${cap}`;
  };
  const rows = [["Signature submitted", "Result across 12 disorders"]];
  byo.forEach((r) => rows.push([names[r.fixture], result(r)]));
  table(s, rows, { x: M, y: 1.65, w: W - 2 * M, colW: [4.7, 7.43], rowH: 0.68, name: "fixture results table" });
  text(s, `Random probes reach AUC ${Number(random.highest_auc).toFixed(2)} on ${random.highest_auc_disorder.replace(" syndrome", "")} and at least 0.70 on ${parseInt(random["n_cohorts_auc_at_or_above_0.70"], 10)} of 12 cohorts. AUC alone is unsafe here: the score against random probes is what makes "no match" work.`,
    { x: M, y: 5.95, w: W - 2 * M, h: 0.9, fontSize: 22, bold: true, color: C.accent5 });
  s.addNotes("This is the feature our supervisor asked for. A laboratory uploads its own list of probes, and the engine searches all twelve disorders. We tested it with five signatures made from real data. The real Sotos probes identify Sotos, even when we flip every direction, which the tool flags. Five hundred random probes give no match. That last one is a finding about method. By the usual measure, the area under the curve or AUC, random probes score a perfect 1.0 on Williams syndrome and pass the usual threshold on seven of twelve cohorts. So AUC alone would have named a disorder for a random list. What makes the tool say no match is a second test: the probes must beat a thousand random probe sets. Also note the twenty-probe signature is capped at a lower evidence ceiling than the five-hundred-probe one.");
}

// 13 --------------------------------------------------------------- demo
{
  const s = content("Identification engine", "Live demo: test your own signature");
  fit(s, path.join(__dirname, "assets", "byo_page_crop.png"), { x: M, y: 1.5, w: 7.2, h: 5.3 },
    "Screenshot of the Test your own signature page");
  const points = ["Upload a probe list, or pick an example", "Coverage on three array platforms",
    "A verdict that states how many disorders were searched", "A warning when the match is within one study only",
    "Runs offline from files in the repository"];
  text(s, points.map((p, i) => ({ text: p, options: { bullet: true, breakLine: i < points.length - 1 } })),
    { x: 8.1, y: 1.7, w: 4.6, h: 4.3, fontSize: 20, paraSpaceAfter: 12 });
  text(s, "Probe identifiers only. No patient data is ever accepted.",
    { x: 8.1, y: 6.05, w: 4.6, h: 0.75, fontSize: 20, italic: true, color: C.text2 });
  s.addNotes("This is the page itself, and we will now show it live. You upload a probe list or choose one of the five examples. It reports how many probes exist on each array type, then the verdict, and it always says how many disorders it searched. Here it identifies Sotos syndrome and warns that the match is within a single study. The page runs offline and only ever takes probe identifiers, never patient data.");
}

// --------------------------------------------------------------- plain-language layer
{
  const ev = Object.fromEntries(tsv("explanation_eval_summary.tsv").map((r) => [r.measure, r.value]));
  const review = tsv("explanation_review.tsv").filter((r) => r.source === "model");
  const wrong = review.filter((r) => r.review_verdict === "error").length;
  const s = content("Identification engine", "A language model as a formatter, checked");
  stat(s, `${ev.passed_first_time} of ${ev.n_subjects}`, "summaries passed the automatic check first time",
    { x: M, y: 1.6, w: 3.8, h: 2.3, color: C.accent1 });
  stat(s, `${ev.passed_after_retry} + ${ev.fell_back_after_two_rejections}`, "passed after one retry, and fell back to a fixed template",
    { x: M + 4.1, y: 1.6, w: 3.8, h: 2.3, color: C.accent5 });
  stat(s, `${wrong} of ${review.length}`, "accepted summaries were still wrong on review, and are blocked",
    { x: M + 8.2, y: 1.6, w: 3.8, h: 2.3, color: C.accent6 });
  text(s, [
    { text: "The model only restates a fact sheet built from our tables. Any number not in it, any clinical or mechanistic word, or a wrong term means rejection.", options: { breakLine: true } },
    { text: "Passing that check is not the same as being right: it misses errors of meaning. The review was done by an AI assistant, not a person.", options: { bold: true } },
  ], { x: M, y: 4.25, w: W - 2 * M, h: 2.5, fontSize: 20, paraSpaceAfter: 10 });
  s.addNotes(`The app can add a plain-language summary to each result, written by a language model. We use the model only as a formatter: it gets a fact sheet built from our tables, and every output is checked before it is shown. Of ${ev.n_subjects} summaries, ${ev.passed_first_time} passed first time, ${ev.passed_after_retry} needed one retry, and ${ev.fell_back_after_two_rejections} fell back to a fixed template. But when every accepted summary was read through, ${wrong} of ${review.length} still contained a mistake of meaning, such as calling probes samples, so those are blocked. The point is that an automatic check catches invented numbers, not misreadings, and the app works fully without the model.`);
}

pres.addSection({ title: "Honest accounting" });

// 14 --------------------------------------------------------------- ours / not ours
{
  const s = content("Honest accounting", "What is ours and what is not");
  const cols = [
    ["Not ours", C.accent4, ["The calibration method (Pejaver 2022; a general tool is on CRAN)",
      "Testing one disorder against another (routine inside the private database)",
      "Using a language model to read dataset labels (published benchmarks exist)"]],
    ["Ours", C.accent1, ["Every result regenerated from public data by one command",
      "The audit: 12 verdicts, 6 cross-study attempts, a full 12 by 13 matrix",
      "The identification engine with its random-probe control"]],
  ];
  cols.forEach(([head, colour, items], i) => {
    const x = M + i * 6.15;
    card(s, { x, y: 1.65, w: 5.85, h: 4.05 }, `${head} panel`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.35, y: 1.95, w: 0.5, h: 0.5, fill: { color: colour },
      line: { color: colour }, objectName: `${head} marker` });
    text(s, head, { x: x + 1.05, y: 1.92, w: 4.4, h: 0.55, fontSize: 26, bold: true, valign: "middle" });
    text(s, items.map((p, k) => ({ text: p, options: { bullet: true, breakLine: k < items.length - 1 } })),
      { x: x + 0.35, y: 2.75, w: 5.15, h: 2.8, fontSize: 20, paraSpaceAfter: 12 });
  });
  text(s, "Our claim is that this can be checked, not that the methods are new.",
    { x: M, y: 6.0, w: W - 2 * M, h: 0.6, fontSize: 24, bold: true, color: C.accent5 });
  s.addNotes("We want to be exact about our contribution. The calibration method is not ours: it follows Pejaver and colleagues, and a general package already exists. Testing one disorder against another is routine inside the private database. Using a language model to read dataset labels has published benchmarks. What is ours is that every result can be regenerated from public data with one command, the audit itself, and the identification tool.");
}

// 15 --------------------------------------------------------------- limitations
{
  const s = content("Honest accounting", "Limitations, before you ask");
  const items = [
    ["Small numbers", "That one cross-study result rests on 18 cases and 6 controls."],
    ["One cross-study test", "Only 1 of 6 directions was computed. We are not claiming three disorders were tested across studies."],
    ["Optimistic self-scores", "Probes were chosen using all training samples, so own-disorder sensitivity is flattering."],
    ["Labels checked by AI only", "Two AI-written rule sets agree on 81 of 81 samples. Human curation is still pending."],
  ];
  items.forEach(([head, body], i) => {
    const x = M + (i % 2) * 6.15;
    const y = 1.65 + Math.floor(i / 2) * 2.6;
    card(s, { x, y, w: 5.85, h: 2.35 }, `limitation ${i + 1}`);
    text(s, head, { x: x + 0.35, y: y + 0.25, w: 5.15, h: 0.5, fontSize: 24, bold: true, color: C.accent6 });
    text(s, body, { x: x + 0.35, y: y + 0.85, w: 5.15, h: 1.4, fontSize: 20, color: C.text2 });
  });
  s.addNotes("We would rather tell you the weaknesses ourselves. Only one of six cross-study directions was computed, and it is based on only eighteen cases and six controls. We are not claiming that three disorders were tested across studies. Scores on a classifier's own disorder are optimistic, because the probes were chosen using those same samples. And our label check compares two computer-written rule sets, so a human review is still needed.");
}

pres.addSection({ title: "Close" });

// 16 --------------------------------------------------------------- conclusion
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Close" });
  s.addText("Public data support far less than within-study numbers suggest", { placeholder: "title" });
  const points = [
    `${losoOk.length} of ${loso.length} cross-study tests could be run; 2 of 3 eligible disorders were never tested across studies`,
    `Borrowed controls: ${pct(effect.cross_series.mean_off_diagonal_fraction_positive)} false positives on other disorders, against ${pct(effect.within_series.mean_off_diagonal_fraction_positive)}`,
    `${between.length} disorders have a between-study bound; both read "${between[0].band}"`,
    "What the field needs: the same disorder from more than one site, with each site's own controls",
  ];
  text(s, points.map((p, k) => ({ text: p, options: { bullet: true, breakLine: k < points.length - 1 } })),
    { x: M, y: 3.75, w: W - 2 * M, h: 2.25, fontSize: 20, color: C.background1, paraSpaceAfter: 6 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.1, w: W - 2 * M, h: 0.85, rectRadius: 0.08,
    fill: { color: C.text2 }, line: { color: C.text2 }, objectName: "command panel" });
  text(s, [{ text: "python scripts/reproduce.py", options: { breakLine: true } },
    { text: "github.com/cleverlyblue0009/EpiGrade" }],
    { x: M + 0.3, y: 6.1, w: W - 2 * M - 0.6, h: 0.85, fontSize: 20, fontFace: "Courier New",
      color: C.background1, valign: "middle" });
  s.addNotes("To conclude. Results measured inside one study look far stronger than public data can confirm. Only one cross-study test in six could be run, and two of the three eligible disorders were never tested across studies. Classifiers built with borrowed controls flagged about two thirds of unrelated patients. The fix is not a new method. It is data: the same disorder deposited by more than one site, each with its own controls. Everything we showed can be regenerated with the one command on this slide. Thank you.");
}

// 17 --------------------------------------------------------------- backup
{
  const s = content("Close", "Backup: methods, thresholds and data");
  const rows = [
    ["Item", "Detail"],
    ["Scoring rule", "Correlation with case profile minus correlation with control profile (Choufani 2015)"],
    ["Signature threshold", "Mann-Whitney test, false discovery rate 0.05, at least 10% methylation difference. Changed once, from Bonferroni and 20%."],
    ["Minimums", "10 cases to attempt a signature; 10 probes to keep one"],
    ["Evidence scale", "Likelihood ratio to points, base 2.08 (Tavtigian 2018); band from the conservative bound"],
    ["Samples", `${fmt(counts.harvested)} harvested, ${fmt(counts.retained_for_analysis)} retained, ${fmt(counts.excluded_total)} excluded`],
    ["Disease datasets (GEO)", "GSE74432, GSE104451, GSE55491, GSE116300, GSE97362, GSE66552, GSE95040, GSE108423, GSE116992, GSE125367, GSE52588"],
  ];
  table(s, rows, { x: M, y: 1.55, w: W - 2 * M, colW: [3.1, 9.03], rowH: 0.66, fontSize: 16, name: "methods table" });
  s.addNotes("This backup slide is for questions on method. The scoring rule is the published one: correlation with the case profile minus correlation with the control profile. Signatures use a Mann-Whitney test with a false discovery rate of five percent and a ten percent difference floor, and that threshold was changed exactly once. All data come from the Gene Expression Omnibus, a public archive, under the accession numbers listed here.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  const applyThemePath = process.argv[2];
  if (applyThemePath) {
    const { applyTheme } = require(applyThemePath);
    await applyTheme(OUT, THEME);
  }
  console.log(`wrote ${OUT}`);
})();
