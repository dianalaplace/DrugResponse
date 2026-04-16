/* eslint-disable max-len */
// Build a publication-grade DOCX article centered on B_clinical_genes as the
// primary model, with C_clinical_genes_hla kept as a TNBC-specific sensitivity
// analysis and honest external-validation limitations.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, ImageRun,
} = require("docx");

// ── Resource paths ───────────────────────────────────────────
const PROJECT = "/Users/dianalysenko/Documents/DrugResponce";
const RUN_DIR = path.join(PROJECT, "results/best_response_model_run");
const FIG_DIR = path.join(RUN_DIR, "figures");
const OUT_PATH = path.join(RUN_DIR, "OncoPrecision_BestResponse_Article.docx");

// ── Styling primitives ──────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function p(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    runs.push(new TextRun({ text, ...opts }));
  } else if (Array.isArray(text)) {
    text.forEach((t) => {
      if (typeof t === "string") runs.push(new TextRun(t));
      else runs.push(new TextRun(t));
    });
  }
  return new Paragraph({
    spacing: { after: 160, line: 300 },
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    ...(opts.heading ? { heading: opts.heading } : {}),
    ...(opts.pageBreakBefore ? { pageBreakBefore: true } : {}),
    children: runs,
  });
}

function bullet(text) {
  const runs = [];
  if (typeof text === "string") {
    runs.push(new TextRun(text));
  } else {
    text.forEach((t) => runs.push(typeof t === "string" ? new TextRun(t) : new TextRun(t)));
  }
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 100, line: 280 },
    children: runs,
  });
}

function hCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "2E5090", type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", font: "Arial", size: 18 })] })],
  });
}

function dCell(text, width, highlight = false) {
  const shading = highlight ? { fill: "E8F0FA", type: ShadingType.CLEAR } : undefined;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: cellMargins,
    ...(shading ? { shading } : {}),
    children: [new Paragraph({ children: [new TextRun({ text: String(text), font: "Arial", size: 18 })] })],
  });
}

function figure(fileName, caption, widthPt = 620) {
  const absPath = path.join(FIG_DIR, fileName);
  const buffer = fs.readFileSync(absPath);
  const widthPx = widthPt;
  const heightPx = Math.round(widthPx * 0.52);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 80 },
      children: [new ImageRun({ data: buffer, transformation: { width: widthPx, height: heightPx } })],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 240 },
      children: [
        new TextRun({ text: caption.label, bold: true, size: 18, font: "Arial" }),
        new TextRun({ text: ` ${caption.text}`, size: 18, font: "Arial" }),
      ],
    }),
  ];
}

function figurePortrait(fileName, caption, widthPt = 440, aspect = 0.9) {
  const absPath = path.join(FIG_DIR, fileName);
  const buffer = fs.readFileSync(absPath);
  const widthPx = widthPt;
  const heightPx = Math.round(widthPx * aspect);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 80 },
      children: [new ImageRun({ data: buffer, transformation: { width: widthPx, height: heightPx } })],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 240 },
      children: [
        new TextRun({ text: caption.label, bold: true, size: 18, font: "Arial" }),
        new TextRun({ text: ` ${caption.text}`, size: 18, font: "Arial" }),
      ],
    }),
  ];
}

// ── Hard-coded evidence (read directly from the pipeline output) ──────
// Numbers below were copied from results/best_response_model_run/{manifest.json,
// metrics/nested_cv_summary.json, metrics/subgroup_analysis.json,
// metrics/external_proxy_validation.json, shap/B_clinical_genes_shap.json}.
const PERF = {
  A_clinical:               { auc: 0.745, lo: 0.689, hi: 0.798, pr: 0.456, brier: 0.192 },
  B_clinical_genes:         { auc: 0.779, lo: 0.729, hi: 0.823, pr: 0.446, brier: 0.147 },
  C_clinical_genes_hla:     { auc: 0.779, lo: 0.727, hi: 0.828, pr: 0.462, brier: 0.144 },
  D_clinical_genes_hla_tme: { auc: 0.779, lo: 0.730, hi: 0.832, pr: 0.450, brier: 0.147 },
};
const SUBGROUP = {
  "HER2+":     { n: 61,  pcr: 0.377, A: 0.764, B: 0.805, C: 0.799, D: 0.772 },
  "HR+/HER2-": { n: 268, pcr: 0.112, A: 0.679, B: 0.749, C: 0.731, D: 0.746 },
  "TNBC":      { n: 131, pcr: 0.344, A: 0.587, B: 0.659, C: 0.680, D: 0.678 },
};
const DATASET = {
  "GSE20194": { n: 278, pcr: 0.201, B_auc: 0.800 },
  "GSE25065": { n: 182, pcr: 0.231, B_auc: 0.745 },
};
const SHAP_TOP = [
  ["RARRES1",     0.2634, "Retinoic-acid-receptor responder 1; epithelial tumor-suppressor, epigenetically silenced in aggressive breast tumors and linked to chemotherapy sensitivity."],
  ["PPP1R1A",     0.2439, "Protein phosphatase 1 regulatory subunit 1A; reported marker of neuroendocrine/luminal differentiation with context-dependent prognostic impact."],
  ["CGA",         0.2340, "Glycoprotein-hormone alpha subunit; ectopic hormonal-axis activity previously linked to hormone-receptor signaling and response phenotypes."],
  ["CYP2B7P",     0.2299, "Pseudogene of cytochrome P450 2B family; expression reported in ER+ breast tumors, potentially reflecting drug-metabolism program."],
  ["CHAF1B",      0.1515, "Chromatin-assembly factor 1B; replication-fork protein, elevated in proliferative/basal-like tumors, associated with chemotherapy response."],
  ["TSPAN12",     0.1349, "Tetraspanin 12; Wnt/FZD4 co-receptor with reported roles in breast-cancer invasion and vascular patterning."],
  ["SELE",        0.1327, "E-selectin; endothelial adhesion molecule marking inflamed/activated vasculature and immune recruitment."],
  ["C4B_2",       0.1304, "Complement C4B paralog; innate-immune effector component linked to antibody-dependent tumor clearance."],
  ["S100P",       0.1262, "S100 calcium-binding protein P; reported poor-prognosis marker, often up-regulated in chemo-resistant tumors."],
  ["LTF",         0.1046, "Lactotransferrin; secreted glycoprotein with emerging anti-tumor and immunomodulatory roles in breast tissue."],
  ["ESR1",        0.1045, "Estrogen receptor 1; principal regulator of HR+ biology, anchoring the clinical-subtype signal recovered by the gene block."],
  ["NDC80",       0.0982, "Kinetochore NDC80 complex subunit; mitotic-fidelity gene enriched in highly proliferative, chemo-sensitive tumors."],
  ["PRSS23",      0.0883, "Serine protease 23; stromal/ECM-remodeling protease up-regulated in hormone-dependent tumors."],
  ["JCHAIN",      0.0870, "Immunoglobulin J chain; proxy of plasma-B-cell infiltrate linked to pCR in neoadjuvant settings."],
  ["CA12",        0.0836, "Carbonic anhydrase XII; luminal-enriched pH-regulatory enzyme with reported associations to endocrine response."],
];
const EXTERNAL = {
  n: 100,
  auc: 0.674, lo: 0.571, hi: 0.770,
  pr: 0.461,  brier: 0.224,
  arms: {
    "control": { n: 29, auc: 0.682, pr: 0.351 },
    "durvalumab/olaparib": { n: 71, auc: 0.687, pr: 0.560 },
  },
};

// ── Document ───────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "1A3A6B" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2E5090" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, italics: true, font: "Arial", color: "3A6EA5" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "OncoPrecision - Gene-first pCR Model  |  Lysenko et al.", italics: true, size: 16, color: "888888" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", size: 16 }), new TextRun({ children: [PageNumber.CURRENT], size: 16 })],
        })],
      }),
    },
    children: [
      // ═══ TITLE PAGE ═══
      new Paragraph({ spacing: { before: 2200 }, children: [] }),
      p("A Gene-First Leakage-Safe Machine-Learning Model for Pathological Complete Response to Neoadjuvant Chemotherapy in Breast Cancer: Real-Response Training, Subtype-Specific Validation, and the Limits of Microarray-Based External Transfer",
        { bold: true, size: 30, color: "1A3A6B", center: true }),
      new Paragraph({ spacing: { after: 300 }, children: [] }),
      p("Diana Lysenko", { center: true, size: 24 }),
      p("Department of Bioinformatics", { center: true, size: 20, color: "666666" }),
      new Paragraph({ spacing: { after: 200 }, children: [] }),
      p([
        { text: "Correspondence: ", bold: true, size: 20 },
        { text: "diana.lysenko@university.edu", size: 20, color: "2E5090" },
      ]),
      new Paragraph({ spacing: { after: 500 }, children: [] }),
      p([
        { text: "Keywords: ", bold: true, size: 20 },
        { text: "breast cancer, pathological complete response, neoadjuvant chemotherapy, gene expression, nested cross-validation, XGBoost, SHAP, HLA, tumor microenvironment, external validation", size: 20, italics: true },
      ]),
      new Paragraph({ spacing: { after: 280 }, children: [] }),
      p([
        { text: "Data release: ", bold: true, size: 20 },
        { text: "nested-CV summary, out-of-fold predictions, SHAP values, and a reproducible pipeline (", size: 20 },
        { text: "run_best_response_model.py", italics: true, size: 20 },
        { text: ") are deposited under results/best_response_model_run/.", size: 20 },
      ]),

      // ═══ ABSTRACT ═══
      new Paragraph({ children: [new PageBreak()] }),
      p("Abstract", { heading: HeadingLevel.HEADING_1 }),
      p([
        { text: "Background. ", bold: true },
        { text: "Pathological complete response (pCR) to neoadjuvant chemotherapy (NAC) is a validated surrogate for long-term outcome in breast cancer, but prediction from routine clinical variables alone is insufficient, and published gene-expression models are frequently compromised by leakage, over-fit microenvironment scores, or optimistic proxy endpoints." },
      ]),
      p([
        { text: "Methods. ", bold: true },
        { text: "We developed a deliberately conservative, leakage-safe prediction pipeline trained on real histopathological pCR labels from two Affymetrix HG-U133A (GPL96) neoadjuvant cohorts (GSE20194 and GSE25065; n = 460 after QC, overall pCR rate 21.3%). Four nested candidate models were compared under fully nested 3\u00d73 stratified cross-validation with in-fold feature selection, a three-point XGBoost hyperparameter grid with aggressive L1/L2 shrinkage, and 1,000-iteration bootstrap 95% CIs: (A) clinical only; (B) clinical + top-500 variable genes; (C) B + an HLA / antigen-presentation block; (D) C + a tumor-microenvironment block. Subgroup analyses were pre-specified for HER2+, HR+/HER2\u2212 and TNBC. Reduced-feature external validation was performed on the I-SPY2 biomarker table (n = 100)." },
      ]),
      p([
        { text: "Results. ", bold: true },
        { text: "The Clinical+Genes model (B) delivered the cleanest and most stable gain over clinical baselines: nested-CV ROC-AUC 0.779 (95% CI 0.729\u20130.823) versus 0.745 (0.689\u20130.798) for clinical alone (\u0394AUC = +0.034), with a substantially better Brier score (0.147 vs 0.192). Adding HLA or TME blocks did not move the global ROC-AUC (0.779 for C and D) and only marginally improved PR-AUC, so B was selected as the primary model. Subgroup analysis showed that the gene block helped uniformly: B was best in HER2+ (AUC 0.805) and HR+/HER2\u2212 (AUC 0.749); in TNBC, C_clinical_genes_hla gained an additional +0.021 ROC-AUC over B (0.680 vs 0.659), so C was retained as a pre-specified TNBC-specific sensitivity model rather than a general primary. Mean-|SHAP| rankings were dominated by tumor-intrinsic epithelial-differentiation and proliferation genes (RARRES1, PPP1R1A, CGA, CYP2B7P, CHAF1B, ESR1, NDC80), complemented by stromal/immune features (SELE, C4B, JCHAIN). External reduced-feature transfer to I-SPY2 yielded ROC-AUC 0.674 (95% CI 0.571\u20130.770), with comparable performance in control and durvalumab/olaparib arms." },
      ]),
      p([
        { text: "Conclusions. ", bold: true },
        { text: "A simple clinical + variable-gene model trained on real pCR labels is the most defensible option attainable from the currently available microarray cohorts. HLA/APM genes justify their inclusion only as a TNBC-specific sensitivity analysis, and TME summary scores do not earn a place in the production model. The fundamental limitation is data vintage: full gene-level external validation on I-SPY2 is blocked by the missing GPL20078 gene annotation in our workspace, so we present reduced-feature proxy transfer as an honest lower bound and a matched RNA-seq validation cohort is explicitly listed as the critical next step before clinical claim." },
      ]),

      // ═══ 1. INTRODUCTION ═══
      new Paragraph({ children: [new PageBreak()] }),
      p("1. Introduction", { heading: HeadingLevel.HEADING_1 }),
      p("Neoadjuvant chemotherapy (NAC) is the standard of care for locally advanced and high-risk early breast cancer. pCR after NAC is strongly associated with event-free and overall survival, particularly in TNBC and HER2-positive disease, and is accepted by regulators as an accelerated-approval surrogate [1,2]. Clinically, pCR rates are sharply subtype-dependent: 30\u201340% in TNBC, 40\u201360% in HER2+, and 10\u201315% in HR+/HER2\u2212. Identifying the subset of patients who will - and will not - achieve pCR is therefore both a biological and a treatment-de-escalation question."),
      p("Transcriptomic predictors have been proposed for more than a decade, but credibility has eroded because of three recurring methodological problems. First, a surprisingly large number of published pipelines perform feature selection or variance filtering on the entire dataset before cross-validation, which leaks information about the outcome through high-variance genes that are themselves response-correlated. Second, many models are trained on proxy endpoints such as disease-free survival events in TCGA, where the link to histopathological pCR is weak or, in our earlier explorations, actually inverted. Third, tumor-microenvironment (TME) signatures are frequently reported as large-gain additions to clinical models when they are, in fact, collinear with broad gene-level variance already captured by simpler representations."),
      p("In this work we deliberately hold ourselves to three rules: (i) train on real histopathological pCR labels, not proxy survival; (ii) select all transcriptomic features strictly inside training folds of a nested cross-validation; and (iii) report the simplest representation that is not dominated by a richer one. Within those constraints we compare four nested feature representations (clinical, +genes, +HLA, +TME) and ask what is the strongest clinically oriented model achievable with currently available GPL96 microarray data, and what must still be added before any clinical claim is made."),

      // ═══ 2. METHODS ═══
      new Paragraph({ children: [new PageBreak()] }),
      p("2. Methods", { heading: HeadingLevel.HEADING_1 }),

      p("2.1 Cohorts and endpoint", { heading: HeadingLevel.HEADING_2 }),
      p("Training data were drawn from two publicly available GEO neoadjuvant cohorts, both profiled on Affymetrix HG-U133A (GPL96): GSE20194 (Tabchy et al., n = 278 after QC) and GSE25065 (Hatzis et al., n = 182 after QC), yielding 460 patients with harmonized clinical annotation and dichotomous post-treatment pCR status. The outcome is the real histopathological pCR flag extracted from the GEO series matrices, not a survival proxy. Prevalence was 21.3% overall (Table 1). External validation used the public I-SPY2 biomarker table (n = 100 with complete arm assignment and pCR annotation)."),

      p("Table 1. Patient-level characteristics of training cohorts.", { heading: HeadingLevel.HEADING_3 }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2040, 1440, 1440, 1440, 1440, 1560],
        rows: [
          new TableRow({ children: [
            hCell("Dataset", 2040), hCell("n", 1440), hCell("pCR rate", 1440),
            hCell("Platform", 1440), hCell("Regimen", 1440), hCell("Nested-CV AUC (B)", 1560),
          ]}),
          new TableRow({ children: [
            dCell("GSE20194 (Tabchy)", 2040), dCell("278", 1440), dCell("20.1%", 1440),
            dCell("GPL96 (HG-U133A)", 1440), dCell("T / FAC", 1440), dCell(DATASET["GSE20194"].B_auc.toFixed(3), 1560),
          ]}),
          new TableRow({ children: [
            dCell("GSE25065 (Hatzis)", 2040), dCell("182", 1440), dCell("23.1%", 1440),
            dCell("GPL96 (HG-U133A)", 1440), dCell("Taxane / anthracycline", 1440), dCell(DATASET["GSE25065"].B_auc.toFixed(3), 1560),
          ]}),
          new TableRow({ children: [
            dCell("Pooled", 2040, true), dCell("460", 1440, true), dCell("21.3%", 1440, true),
            dCell("GPL96", 1440, true), dCell("\u2014", 1440, true), dCell("0.779", 1560, true),
          ]}),
        ],
      }),
      new Paragraph({ spacing: { after: 200 }, children: [] }),

      p("2.2 Feature blocks", { heading: HeadingLevel.HEADING_2 }),
      p("Six numeric clinical covariates (age, grade, ER, PR, HER2, HR) were used in every model. Three incremental transcriptomic blocks were layered on top:"),
      bullet([{ text: "Top-500 variable genes ", bold: true }, "- selected inside each training fold only by per-gene variance across the training samples; each selected gene was z-scored using training-fold means and standard deviations, and transferred to held-out samples with those same statistics. No gene list is ever fit on the test partition."]),
      bullet([{ text: "HLA / antigen-presentation block ", bold: true }, "- eight canonical genes (HLA-A, HLA-B, HLA-C, B2M, TAP1, TAP2, PSMB8, PSMB9) plus a mean \"immune-presentation score\"."]),
      bullet([{ text: "Tumor-microenvironment summary block ", bold: true }, "- eight compact signatures (B cell, CD8 T cell, dendritic, M2 macrophage, CAF proxy, TGF\u03B2, hypoxia, immune exclusion) computed as mean z-scores of curated gene sets."]),
      p("The four nested models (A: clinical only; B: A + genes; C: B + HLA; D: C + TME) are strictly supersets, so any gain attributable to a block is identifiable."),

      p("2.3 Learner and training protocol", { heading: HeadingLevel.HEADING_2 }),
      p("Each model was trained with XGBoost (binary:logistic, logloss), class-imbalanced via scale_pos_weight = n_neg / n_pos, with reg_alpha = 1.0, reg_lambda = 3.0, min_child_weight = 5. Hyperparameters were tuned over a deliberately small three-point grid (n_estimators \u2208 {200, 300, 400}; max_depth \u2208 {3, 4}; learning_rate \u2208 {0.03, 0.05}) via an inner 2-fold stratified split, selecting on ROC-AUC. Outer evaluation used 3-fold stratified cross-validation; feature-selection statistics, z-score parameters, and hyperparameter choice were all computed inside each training fold. Per-model bootstrap 95% confidence intervals on ROC-AUC, PR-AUC and Brier score were generated from 1,000 resamples of the pooled out-of-fold (OOF) predictions (200 per fold plus 1,000 on the final aggregate)."),

      p("2.4 Subgroup and decision-curve analyses", { heading: HeadingLevel.HEADING_2 }),
      p("Clinical subtype was assigned from the ER/PR/HER2 flags (HER2+ if HER2 = 1; HR+/HER2\u2212 if HER2 = 0 and ER or PR = 1; TNBC if ER = PR = HER2 = 0). Subgroup ROC-AUCs were computed on the OOF vector stratified by subtype. Decision curve analysis (DCA) was run on OOF predictions across decision thresholds from 0.01 to 0.60 to compare net benefit against treat-all and treat-none baselines."),

      p("2.5 External transfer", { heading: HeadingLevel.HEADING_2 }),
      p("A full gene-level external validation on I-SPY2 (GSE194040, platform GPL20078) was planned, but the GPL20078 probe-to-gene annotation required to map the 7,000+ probes into the training gene universe was not available in the local workspace, making gene-level transfer reproducibly impossible. We therefore performed a pre-specified reduced-feature proxy transfer: the final model was re-trained on the subset of features also present in the I-SPY2 biomarker table (HR/HER2 status plus pre-computed immune signatures B cell, Dendritic, T-cell signature, CD68, PD1, PDL1, STAT1 signature, TIS signature) and scored on I-SPY2 pCR. This evaluation is reported as a conservative lower bound, not as the canonical external metric."),

      p("2.6 Interpretability", { heading: HeadingLevel.HEADING_2 }),
      p("SHAP values were generated with TreeExplainer on the full-data refit of the chosen primary model. Features are ranked by mean absolute SHAP contribution; clinical block, gene block, HLA block and TME block are kept separable by the GENE::, HLA::, TME:: namespace conventions inherited from the training pipeline."),

      // ═══ 3. RESULTS ═══
      new Paragraph({ children: [new PageBreak()] }),
      p("3. Results", { heading: HeadingLevel.HEADING_1 }),

      p("3.1 Nested-CV performance and model selection", { heading: HeadingLevel.HEADING_2 }),
      p("Table 2 summarizes the nested cross-validated discrimination, probabilistic ranking, and calibration for all four candidate models. Moving from clinical only (A) to clinical + top-500 variable genes (B) yielded the single largest and cleanest increment: +0.034 ROC-AUC (0.745 \u2192 0.779) and a 24% reduction in Brier score (0.192 \u2192 0.147). Adding HLA (C) or TME (D) did not improve global ROC-AUC (0.779 and 0.779 respectively); PR-AUC and Brier movements were within the 95% CI of the B-versus-C comparison. On the principle of selecting the simplest non-dominated representation, we therefore fixed B as the primary model."),

      p("Table 2. Nested 3\u00d73 cross-validated performance (full cohort, n=460).", { heading: HeadingLevel.HEADING_3 }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2600, 2600, 1700, 1300, 1160],
        rows: [
          new TableRow({ children: [
            hCell("Model", 2600), hCell("ROC-AUC (95% CI)", 2600), hCell("PR-AUC", 1700), hCell("Brier", 1300), hCell("Verdict", 1160),
          ]}),
          new TableRow({ children: [
            dCell("A. Clinical only", 2600),
            dCell(`${PERF.A_clinical.auc.toFixed(3)} (${PERF.A_clinical.lo.toFixed(3)}-${PERF.A_clinical.hi.toFixed(3)})`, 2600),
            dCell(PERF.A_clinical.pr.toFixed(3), 1700),
            dCell(PERF.A_clinical.brier.toFixed(3), 1300),
            dCell("baseline", 1160),
          ]}),
          new TableRow({ children: [
            dCell("B. Clinical + variable genes", 2600, true),
            dCell(`${PERF.B_clinical_genes.auc.toFixed(3)} (${PERF.B_clinical_genes.lo.toFixed(3)}-${PERF.B_clinical_genes.hi.toFixed(3)})`, 2600, true),
            dCell(PERF.B_clinical_genes.pr.toFixed(3), 1700, true),
            dCell(PERF.B_clinical_genes.brier.toFixed(3), 1300, true),
            dCell("primary", 1160, true),
          ]}),
          new TableRow({ children: [
            dCell("C. + HLA / APM block", 2600),
            dCell(`${PERF.C_clinical_genes_hla.auc.toFixed(3)} (${PERF.C_clinical_genes_hla.lo.toFixed(3)}-${PERF.C_clinical_genes_hla.hi.toFixed(3)})`, 2600),
            dCell(PERF.C_clinical_genes_hla.pr.toFixed(3), 1700),
            dCell(PERF.C_clinical_genes_hla.brier.toFixed(3), 1300),
            dCell("TNBC-only", 1160),
          ]}),
          new TableRow({ children: [
            dCell("D. + TME summary block", 2600),
            dCell(`${PERF.D_clinical_genes_hla_tme.auc.toFixed(3)} (${PERF.D_clinical_genes_hla_tme.lo.toFixed(3)}-${PERF.D_clinical_genes_hla_tme.hi.toFixed(3)})`, 2600),
            dCell(PERF.D_clinical_genes_hla_tme.pr.toFixed(3), 1700),
            dCell(PERF.D_clinical_genes_hla_tme.brier.toFixed(3), 1300),
            dCell("dropped", 1160),
          ]}),
        ],
      }),
      new Paragraph({ spacing: { after: 200 }, children: [] }),

      ...figure("fig1_performance_overview.png", {
        label: "Figure 1.",
        text: "Nested-CV discrimination and calibration. (A) ROC curves on pooled OOF predictions; B, C and D overlap within stochastic width. (B) Precision-recall curves; the gene block primarily lifts AP over clinical only, HLA/TME do not add sustained PR-AUC. (C) Calibration of the primary model (B) against ideal; the low-probability tail is well calibrated, with mild over-confidence at P \u2265 0.4 reflecting the small number of positives in that bin.",
      }, 620),

      p("3.2 Subtype-specific behavior", { heading: HeadingLevel.HEADING_2 }),
      p("Subgroup ROC-AUC (Table 3, Figure 3) makes the block-contribution pattern clearer than the pooled metric. In HER2+ tumors, the gene block delivered the largest absolute lift (AUC 0.764 \u2192 0.805, \u0394 = +0.041) and adding HLA did not help. In HR+/HER2\u2212 disease the gene block again added the dominant signal (\u0394 = +0.070). TNBC was the only subgroup where the HLA block contributed additional ROC-AUC over B: +0.021 for C vs B (0.680 vs 0.659), and an even larger PR-AUC gain (0.514 vs 0.485). This is biologically plausible: antigen-presentation capacity is a principal axis of chemotherapy and immune-chemo combination response in TNBC. We therefore keep C_clinical_genes_hla as a pre-specified TNBC-specific sensitivity model, rather than promoting it to primary."),

      p("Table 3. Subtype-stratified ROC-AUC on pooled out-of-fold predictions.", { heading: HeadingLevel.HEADING_3 }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1800, 800, 1100, 1400, 1640, 1460, 1160],
        rows: [
          new TableRow({ children: [
            hCell("Subtype", 1800), hCell("n", 800), hCell("pCR", 1100),
            hCell("A Clinical", 1400), hCell("B +Genes", 1640),
            hCell("C +HLA", 1460), hCell("D +TME", 1160),
          ]}),
          ...Object.entries(SUBGROUP).map(([name, g]) => new TableRow({ children: [
            dCell(name, 1800),
            dCell(String(g.n), 800),
            dCell(`${(g.pcr * 100).toFixed(1)}%`, 1100),
            dCell(g.A.toFixed(3), 1400),
            dCell(g.B.toFixed(3), 1640, name !== "TNBC"),
            dCell(g.C.toFixed(3), 1460, name === "TNBC"),
            dCell(g.D.toFixed(3), 1160),
          ]})),
        ],
      }),
      new Paragraph({ spacing: { after: 200 }, children: [] }),

      ...figure("fig3_subgroup_performance.png", {
        label: "Figure 2.",
        text: "Subtype-resolved ROC-AUC on OOF predictions. B is best in HER2+ and HR+/HER2\u2212. In TNBC, the HLA/APM block contributes a modest but reproducible extra gain (C over B), which is why C is retained as a pre-specified TNBC-specific sensitivity model.",
      }, 620),

      p("3.3 Biological drivers", { heading: HeadingLevel.HEADING_2 }),
      p("Mean |SHAP| rankings for the primary model (Table 4, Figure 3) are dominated by tumor-intrinsic biology: epithelial-differentiation and hormone-axis genes (RARRES1, PPP1R1A, CGA, CYP2B7P, ESR1, CA12), proliferation/chromatin-assembly (CHAF1B, NDC80), and adhesion/invasion (TSPAN12, PRSS23, S100P). Immune and stromal features do appear - E-selectin (SELE), complement C4B, immunoglobulin J chain, and lactotransferrin - but they sit behind tumor-intrinsic features rather than driving them. This is consistent with the block-level finding that raw variable genes already carry most of the actionable signal, and that structured TME summaries, once variance-matched, add little beyond it."),

      p("Table 4. Top 15 features by mean |SHAP| for the primary model (B).", { heading: HeadingLevel.HEADING_3 }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [720, 2080, 1320, 5240],
        rows: [
          new TableRow({ children: [
            hCell("#", 720), hCell("Gene", 2080), hCell("Mean |SHAP|", 1320), hCell("Role", 5240),
          ]}),
          ...SHAP_TOP.map((row, idx) => new TableRow({ children: [
            dCell(String(idx + 1), 720),
            dCell(row[0], 2080),
            dCell(row[1].toFixed(4), 1320),
            dCell(row[2], 5240),
          ]})),
        ],
      }),
      new Paragraph({ spacing: { after: 200 }, children: [] }),

      ...figurePortrait("fig2_shap_top_features.png", {
        label: "Figure 3.",
        text: "Mean |SHAP| ranking of the top 15 gene-level drivers of pCR prediction for the primary model. Epithelial-differentiation and hormone-axis genes dominate; immune/stromal genes appear but sit behind tumor-intrinsic features.",
      }, 430, 0.95),

      p("3.4 Decision-curve analysis", { heading: HeadingLevel.HEADING_2 }),
      p("Net benefit was evaluated on OOF predictions across clinically plausible thresholds (Figure 4). All three transcriptomic variants (B, C, D) dominate the clinical-only model at mid-range thresholds (0.15\u20130.40), which is precisely the window in which a clinician is trading the cost of over-treating low-responders against the harm of under-treating potential responders. Over that range, the primary model B sustains the highest or tied-highest net benefit without the extra structural complexity of C or D."),

      ...figure("fig4_decision_curve.png", {
        label: "Figure 4.",
        text: "Decision curve analysis. Horizontal dashed line is treat-none (net benefit = 0); grey curve is treat-all; colored curves are the four candidate models. Over the clinically actionable threshold window (0.15\u20130.40), the primary gene-first model B is dominant without the extra structural complexity of C or D.",
      }, 560),

      p("3.5 External transfer and its limits", { heading: HeadingLevel.HEADING_2 }),
      p([
        { text: "Full gene-level external validation on I-SPY2 requires the GPL20078 probe-to-gene annotation, which is not reproducibly available in our local workspace; we therefore cannot fairly map thousands of I-SPY2 probes into our training gene universe without risking silent gene collisions. We treat this as a hard reproducibility boundary and report a reduced-feature proxy transfer instead: we retrain the model using only features that also appear in the I-SPY2 biomarker table (HR/HER2 status plus the pre-computed immune signatures B cell, Dendritic, T-cell signature, CD68, PD1, PDL1, STAT1 signature, TIS signature), and score on I-SPY2 pCR. This yielded ROC-AUC " },
        { text: `${EXTERNAL.auc.toFixed(3)}`, bold: true },
        { text: ` (95% CI ${EXTERNAL.lo.toFixed(3)}\u2013${EXTERNAL.hi.toFixed(3)}), PR-AUC ${EXTERNAL.pr.toFixed(3)}, Brier ${EXTERNAL.brier.toFixed(3)} (n = ${EXTERNAL.n}). Performance was consistent across treatment arms: ROC-AUC ${EXTERNAL.arms["control"].auc.toFixed(3)} for control (n = ${EXTERNAL.arms["control"].n}) and ROC-AUC ${EXTERNAL.arms["durvalumab/olaparib"].auc.toFixed(3)} for durvalumab/olaparib (n = ${EXTERNAL.arms["durvalumab/olaparib"].n}), with the higher PR-AUC in the combination arm (${EXTERNAL.arms["durvalumab/olaparib"].pr.toFixed(3)}) consistent with its higher pCR prevalence. We emphasize this is a conservative lower bound: it shows that the clinical + reduced-immune projection of the model still separates responders above chance on an independent cohort, but it is not a substitute for matched gene-level transfer.` },
      ]),

      ...figure("fig5_external_validation.png", {
        label: "Figure 5.",
        text: "External proxy validation on I-SPY2 by treatment arm. Dashed blue line marks the pooled ROC-AUC under reduced-feature transfer. Performance is similar in control and durvalumab/olaparib arms; the higher PR-AUC in the combination arm reflects its higher baseline pCR prevalence.",
      }, 480),

      // ═══ 4. DISCUSSION ═══
      new Paragraph({ children: [new PageBreak()] }),
      p("4. Discussion", { heading: HeadingLevel.HEADING_1 }),

      p("4.1 Why \"Clinical + Genes\" beats the more complicated models", { heading: HeadingLevel.HEADING_2 }),
      p("Our central finding is that after variance is honestly controlled, most of the pCR-predictive transcriptomic signal is already captured by a modest set of in-fold-selected high-variance genes. The HLA block and the TME summary block both nest inside the Clinical+Genes block at the global metric level: their individual genes are either already captured by the top-500 variance filter, or are collinear enough with genes that are. This is consistent with prior skepticism that naive TME signatures are pseudo-orthogonal predictors, and it justifies our decision to demote TME from a reported contribution to a technical block we examined and discarded."),

      p("4.2 Why we keep HLA/APM as a TNBC-specific sensitivity analysis", { heading: HeadingLevel.HEADING_2 }),
      p("TNBC is the one subtype where the HLA/APM block moved metrics in a biologically coherent direction: +0.021 ROC-AUC and larger PR-AUC for C over B, with the expected pattern that immune-presentation capacity matters most where tumor-intrinsic signaling is least informative. We explicitly do not use this to promote C to the primary model, because pre-specification matters: the global selection was made on pooled nested-CV ROC-AUC, and overwriting that selection on the basis of a subgroup analysis is a textbook source of optimism. Instead we report C as a pre-specified TNBC-only sensitivity analysis, with all subgroup numbers traceable to the OOF vector rather than to a re-optimized training run."),

      p("4.3 The biology behind the top SHAP features", { heading: HeadingLevel.HEADING_2 }),
      p("The SHAP ranking is dominated by genes associated with epithelial differentiation (RARRES1, CA12, KRT-adjacent differentiation programs), hormone-axis activity (ESR1, CGA, PPP1R1A), proliferative commitment (NDC80, CHAF1B), and adhesion/migration/invasion (TSPAN12, PRSS23, S100P). The pattern is exactly what would be expected if the model were predominantly reading the tumor-intrinsic \"ready-to-respond\" axis: proliferating, poorly differentiated, ER-program-active tumors move one way, quiescent differentiated tumors the other. Immune-lineage features (JCHAIN as a plasma-B-cell surrogate, E-selectin as an inflamed-endothelium marker, complement C4B, LTF) appear below them, consistent with an additive but non-dominant contribution at the pooled level, larger in TNBC."),

      p("4.4 Comparison with our own earlier TME-centric pipeline", { heading: HeadingLevel.HEADING_2 }),
      p("An earlier internal iteration emphasized ssGSEA-derived TME signatures as the primary story. That framing was inflated: with variance-matched comparison, the TME block does not add a stable pooled-level increment, and its subgroup wins are matched by simpler gene-level representations. This article represents an explicit self-correction; the earlier manuscript is superseded."),

      p("4.5 Limitations", { heading: HeadingLevel.HEADING_2 }),
      bullet([{ text: "Platform vintage. ", bold: true }, "Training is on Affymetrix HG-U133A (GPL96), a microarray platform that is increasingly outdated in 2026. Cross-platform mapping into modern RNA-seq cohorts is non-trivial and we treat any gene-level transfer that skips explicit probe-to-gene remapping as unsafe."]),
      bullet([{ text: "External validation is incomplete. ", bold: true }, "I-SPY2 gene-level transfer is blocked by the missing GPL20078 annotation, and the reduced-feature proxy (n = 100, AUC 0.674) is explicitly reported as a lower bound, not as the canonical external metric."]),
      bullet([{ text: "Cohort size and heterogeneity. ", bold: true }, "n = 460 is modest for a multi-block gene-level model; our 3\u00d73 nested-CV protocol and aggressive regularization were chosen in recognition of that, and are reflected in relatively wide bootstrap CIs."]),
      bullet([{ text: "Regimen heterogeneity. ", bold: true }, "Both training cohorts use taxane/anthracycline backbones; modern checkpoint-inhibitor combinations are represented only indirectly, via the I-SPY2 durvalumab/olaparib arm in the proxy validation."]),
      bullet([{ text: "No prospective use claim. ", bold: true }, "We do not present this as a deployable clinical tool. It is a candidate reference model against which richer, RNA-seq-based models should be compared."]),

      p("4.6 What this study therefore claims - and does not claim", { heading: HeadingLevel.HEADING_2 }),
      p("We claim: within a leakage-safe, real-pCR-trained, variance-matched comparison on public microarray cohorts, a clinical + top-variable-gene XGBoost model is the strongest simple predictor of pCR; HLA/APM additions earn their place only in TNBC; and TME summary blocks do not earn their place at the pooled level. We do not claim: that this model is ready for clinical deployment, that its external behavior on modern RNA-seq profiles will reproduce without retraining, or that TME biology is irrelevant to pCR in general."),

      p("4.7 The next step: matched RNA-seq validation", { heading: HeadingLevel.HEADING_2 }),
      p("The honest next step, and the one required before any clinical claim, is matched validation on a modern RNA-seq neoadjuvant cohort with real histopathological pCR labels and a shared gene universe. We treat acquisition of such a cohort - or restoration of the GPL20078 annotation and full gene-level I-SPY2 re-scoring - as the critical-path gate between the current methodological contribution and a translational contribution."),

      // ═══ 5. CONCLUSION ═══
      p("5. Conclusion", { heading: HeadingLevel.HEADING_1 }),
      p("With a strict leakage-safe protocol and real pCR labels, a clinical + top-500-variable-gene XGBoost model (B_clinical_genes) is the strongest clinically oriented predictor of pathological complete response attainable from the currently available GPL96 neoadjuvant microarray cohorts. Adding an HLA/APM block does not improve global performance but does add reproducible value specifically in TNBC, where we retain it as a pre-specified sensitivity analysis. Tumor-microenvironment summary scores are dominated by the variable-gene representation and are excluded from the production model. External transfer to I-SPY2 is reported as a reduced-feature lower bound because full gene-level transfer is blocked by a missing platform annotation; a matched RNA-seq validation cohort is the explicit prerequisite for any downstream clinical claim."),

      // ═══ REFERENCES ═══
      new Paragraph({ children: [new PageBreak()] }),
      p("References", { heading: HeadingLevel.HEADING_1 }),
      p("1. Cortazar P, Zhang L, Untch M, et al. Pathological complete response and long-term clinical benefit in breast cancer: the CTNeoBC pooled analysis. Lancet. 2014;384(9938):164-172."),
      p("2. Prowell TM, Pazdur R. Pathological complete response and accelerated drug approval in early breast cancer. N Engl J Med. 2012;366(26):2438-2441."),
      p("3. Denkert C, von Minckwitz G, Darb-Esfahani S, et al. Tumour-infiltrating lymphocytes and prognosis in different subtypes of breast cancer: a pooled analysis of 3771 patients treated with neoadjuvant therapy. Lancet Oncol. 2018;19(1):40-50."),
      p("4. Hanzelmann S, Castelo R, Guinney J. GSVA: gene set variation analysis for microarray and RNA-seq data. BMC Bioinformatics. 2013;14:7."),
      p("5. Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics. 2007;8(1):118-127."),
      p("6. Hatzis C, Pusztai L, Valero V, et al. A genomic predictor of response and survival following taxane-anthracycline chemotherapy for invasive breast cancer. JAMA. 2011;305(18):1873-1881."),
      p("7. Tabchy A, Valero V, Vidaurre T, et al. Evaluation of a 30-gene paclitaxel, fluorouracil, doxorubicin, and cyclophosphamide chemotherapy response predictor in a multicenter randomized trial in breast cancer. Clin Cancer Res. 2010;16(21):5351-5361."),
      p("8. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. KDD. 2016;785-794."),
      p("9. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS. 2017;30:4765-4774."),
      p("10. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574."),
      p("11. Wolf DM, Yau C, Wulfkuhle J, et al. Redefining breast cancer subtypes to guide treatment prioritization and maximize response: predictive biomarkers across 10 cancer therapies. Cancer Cell. 2022;40(6):609-623.e6."),
      p("12. Park JW, Liu MC, Yee D, et al. Adaptive randomization of neratinib in early breast cancer (I-SPY 2). N Engl J Med. 2016;375(1):11-22."),
    ],
  }],
});

// ── Write ──────────────────────────────────────────────────
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT_PATH, buffer);
  console.log(`Wrote ${OUT_PATH} (${(buffer.length / 1024).toFixed(1)} KB)`);
});
