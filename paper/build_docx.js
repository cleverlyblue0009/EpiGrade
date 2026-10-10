// Build paper/EpiGrade_paper.docx from paper/EpiGrade_paper.md.
// Usage: NODE_PATH=<dir with docx installed>/node_modules node paper/build_docx.js
// The Markdown file is the source of truth; this only lays it out as a Word document.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, LevelFormat, Footer, PageNumber,
} = require("docx");

const here = __dirname;
const lines = fs.readFileSync(path.join(here, "EpiGrade_paper.md"), "utf8").split(/\r?\n/);
const FONT = "Times New Roman";
const BODY = 22; // half-points
const CONTENT_WIDTH = 9026; // A4 with 1-inch margins, in DXA

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// Inline Markdown: **bold**, *italic*, `code`, $math$ (rendered as italic with _x as subscript).
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$)/g;
  let last = 0;
  let m;
  const push = (t, extra = {}) => {
    if (t) out.push(new TextRun({ text: t, font: FONT, size: BODY, ...base, ...extra }));
  };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { font: "Consolas", size: BODY - 2 });
    else if (tok.startsWith("$")) {
      const parts = tok.slice(1, -1).split(/_(\w)/);
      parts.forEach((p, i) => push(p, { italics: true, subScript: i % 2 === 1 }));
    } else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out;
}

const para = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}),
  spacing: { after: 140, line: 300 },
  alignment: opts.alignment || AlignmentType.JUSTIFIED,
  ...opts.para,
});

function table(rows) {
  const n = rows[0].length;
  const longest = Array.from({ length: n }, (_, j) => Math.max(...rows.map((r) => r[j].length)));
  const weight = longest.map((l) => Math.min(Math.max(l, 8), 60));
  const total = weight.reduce((a, b) => a + b, 0);
  const widths = weight.map((w) => Math.floor((w / total) * CONTENT_WIDTH));
  widths[n - 1] += CONTENT_WIDTH - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((cells, i) => new TableRow({
      tableHeader: i === 0,
      children: cells.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        shading: i === 0 ? { type: ShadingType.CLEAR, fill: "E8EEF7" } : undefined,
        borders: { top: border, bottom: border, left: border, right: border },
        margins: { top: 60, bottom: 60, left: 90, right: 90 },
        children: [new Paragraph({
          children: runs(c, { size: BODY - 3, bold: i === 0 }),
          alignment: AlignmentType.LEFT,
        })],
      })),
    })),
  });
}

const children = [];
let numberedLists = 0; // each numbered list gets its own reference so it restarts at 1
let inNumberedList = false;
let i = 0;
while (i < lines.length) {
  const line = lines[i];
  if (!line.trim() || line.trim() === "---") { i += 1; continue; }

  if (line.startsWith("# ")) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun({ text: line.slice(2), font: FONT, size: 34, bold: true })],
    }));
  } else if (line.startsWith("## ") || line.startsWith("### ")) {
    const level = line.startsWith("### ") ? 2 : 1;
    children.push(new Paragraph({
      heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
      spacing: { before: level === 1 ? 320 : 220, after: 120 },
      keepNext: true,
      children: [new TextRun({
        text: line.replace(/^#+ /, ""), font: FONT, size: level === 1 ? 28 : 24, bold: true,
        color: "000000",
      })],
    }));
  } else if (line.startsWith("|")) {
    const block = [];
    while (i < lines.length && lines[i].startsWith("|")) { block.push(lines[i]); i += 1; }
    const rows = block.filter((r) => !/^\|[\s|:-]+\|$/.test(r))
      .map((r) => r.slice(1, -1).split("|").map((c) => c.trim()));
    children.push(table(rows));
    children.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
    continue;
  } else if (line.startsWith("![")) {
    const m = line.match(/^!\[(.*)\]\((.*)\)$/);
    const file = path.resolve(here, m[2]);
    const { w, h } = pngSize(file);
    const width = 600; // pixels at 96 dpi: the full text width
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(file),
        transformation: { width, height: Math.round((width * h) / w) },
        altText: { title: m[1], description: m[1], name: path.basename(file) },
      })],
    }));
    children.push(para(m[1], {
      alignment: AlignmentType.CENTER, run: { size: BODY - 3, italics: true },
    }));
  } else if (line.startsWith("$$")) {
    const formula = line.replace(/\$\$/g, "").trim();
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 100, after: 160 },
      children: runs(`$${formula}$`),
    }));
  } else if (/^\d+\. /.test(line) || line.startsWith("- ")) {
    const numbered = /^\d+\. /.test(line);
    if (numbered && !inNumberedList) numberedLists += 1;
    inNumberedList = numbered;
    children.push(new Paragraph({
      numbering: { reference: numbered ? `numbers${numberedLists}` : "bullets", level: 0 },
      spacing: { after: 80, line: 290 },
      children: runs(line.replace(/^(\d+\.|-) /, "")),
    }));
  } else if (/^\[\d+\] /.test(line)) {
    children.push(para(line, {
      alignment: AlignmentType.LEFT, run: { size: BODY - 2 },
      para: { indent: { left: 500, hanging: 500 }, spacing: { after: 90 } },
    }));
  } else if (line.startsWith("**Upasana") || line.startsWith("Vellore Institute")) {
    children.push(para(line, { alignment: AlignmentType.CENTER }));
  } else {
    children.push(para(line));
  }
  if (!/^\d+\. /.test(line)) inNumberedList = false;
  i += 1;
}

const level = (format, text) => [{
  level: 0, format, text, alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 620, hanging: 320 } } },
}];

const doc = new Document({
  creator: "U. Bhaumik, S. Saravanakumar, D. Tyagi",
  title: "EpiGrade: An Open, Reproducible Benchmark for DNA Methylation Episignature Classifiers",
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  numbering: {
    config: [
      ...Array.from({ length: Math.max(numberedLists, 1) }, (_, k) => ({
        reference: `numbers${k + 1}`, levels: level(LevelFormat.DECIMAL, "%1."),
      })),
      { reference: "bullets", levels: level(LevelFormat.BULLET, "•") },
    ],
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = path.join(here, "EpiGrade_paper.docx");
  fs.writeFileSync(out, buffer);
  console.log(`wrote ${out} (${(buffer.length / 1e6).toFixed(1)} MB)`);
});
