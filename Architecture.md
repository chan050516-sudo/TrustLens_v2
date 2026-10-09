# TrustLens v2

**Document Forensics Through Multi-Engine Evidence Fusion**

TrustLens is a document forensics system that determines whether a business or legal document (invoice, payslip, bank statement, contract, certificate, resume, etc.) has been forged, altered, or misrepresented.

It runs **five independent forensic engines** over a shared physical representation of the document, then fuses their outputs through a reasoning VLM that plays the role of a detective.

> **No single engine decides. Evidence accumulates. A detective reasons.**

---

## Table of Contents

1. [Why TrustLens](#why-trustlens)
2. [System Architecture](#system-architecture)
3. [Cross-Layer Design Principles](#cross-layer-design-principles)
4. [Engine Architectures](#engine-architectures)
   - [Perception](#1-perception)
   - [Metadata](#2-metadata)
   - [Visual](#3-visual)
   - [Reconciliation](#4-reconciliation)
   - [Grounding](#5-grounding)
   - [Semantic](#6-semantic)
   - [Detective](#7-detective)
5. [Walkthrough](#walkthrough)
6. [Project Structure](#project-structure)
7. [Tech Stack](#tech-stack)
8. [Status & Roadmap](#status--roadmap)
9. [Philosophy](#philosophy)

---

## Why TrustLens

Traditional document verification relies on a single signal — a metadata check, a visual inspection, or an arithmetic audit. Real forgery rarely fails on just one axis. A forged payslip might have:

- clean metadata (produced by a common PDF library),
- clean typography (rendered in a single pass),
- **but** a running balance that doesn't close,
- **and** a net pay that doesn't equal gross minus deductions,
- **and** an EPF rate that matches no statutory schedule.

No single engine would flag this. TrustLens would.

The system does not output a risk score. It outputs a **case**: what was noticed, why it matters, and where in the document it can be found.

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         ForensicPipeline                              │
│                    (top-level orchestrator)                           │
└───────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌────────────────┐      ┌──────────────────┐       ┌───────────────────┐
│  Perception    │      │  5 Engines       │       │   Detective       │
│                │      │  (parallel)      │       │                   │
│  • DocumentIR  │      │                  │       │  1. CaseFile      │
│  • DTO IR      │      │  1. Metadata     │       │  2. Render        │
│  • Annotated   │      │  2. Visual       │       │  3. VLM reason    │
│    images      │      │  3. Reconciliation│      │  4. Risk report   │
│                │      │  4. Grounding    │       │                   │
│                │      │  5. Semantic     │       │                   │
└────────────────┘      └──────────────────┘       └───────────────────┘
        │                         │                         ▲
        │                         ▼                         │
        │                 ┌─────────────────┐               │
        └────────────────▶│ Evidence +      │───────────────┘
                          │ Contexts        │
                          └─────────────────┘
```

### Two Roles, Seven Layers

| Layer | Role | Primary Output |
|---|---|---|
| **Perception** | Physical extraction | `DocumentIR`, `ReconciliationDTOIR`, `GroundingDTOIR`, annotated pages |
| **Metadata** | File container forensics | `MetadataContext` + Evidence |
| **Visual** | Page content forensics | `VisualContext` + Evidence |
| **Reconciliation** | Internal arithmetic | `ReconciliationContext` + Evidence |
| **Grounding** | External world | `GroundingContext` (no Evidence) |
| **Semantic** | Meaning-level analysis | Evidence |
| **Detective** | Cross-source reasoning | `DetectiveReport` |

---

## Cross-Layer Design Principles

These principles cut **across every engine**. They are the architectural spine of TrustLens.

### 1. Observation ID as the Universal Anchor

Every physical line of text receives a globally-unique ID:

```
observation_id = page * 1000 + local_index
```

This single integer flows through **every** layer:

- OCR assigns it.
- Docling regions are matched to it.
- Tables reference cells via it.
- DTO IR VLM extraction cites it.
- Every Evidence produced by every engine carries it.
- Annotated images render it directly on top of the source text.

**Consequence**: any claim made by any engine — down to the smallest metadata anomaly — can be traced back to a specific rectangle in a specific page. There is no layer where traceability is lost.

### 2. Dual-Channel Output: Evidence + Context

Every engine produces **two parallel streams**:

- **Evidence** — machine-checkable facts, e.g. `RECONCILIATION_RUNNING_BALANCE_MISMATCH`.
- **Context** — structured observations that require reasoning, e.g. an unusual XMP history chain.

Evidence drives the annotated summary. Context feeds the Detective. This avoids two common failure modes:

- boiling everything down to numeric scores (loses nuance),
- drowning the reasoning model in raw data (loses signal).

Grounding deliberately breaks this pattern — it produces *only* Context, never Evidence — because "external lookup" is inherently observation, not conclusion.

### 3. Graceful Degradation

Every engine runs inside its own try/except at the orchestration layer. If Metadata fails, Visual still runs. If a Visual VLM call times out, other engines are unaffected. Failures are collected in a top-level `errors` list and surfaced to the Detective — which is explicitly told which engines actually ran.

Nothing crashes the pipeline. Nothing is silently skipped.

### 4. Structural Compression, Semantic Preservation

Where information is *semantically* meaningful (identity fields, timestamps, source labels), it is preserved exactly. Where information is *structurally* redundant (repeated observation_id lists, repeated source labels), it is compressed:

```
[1026, 1027, 1028, 1045, 1046]  →  ["1026-1028", "1045-1046"]
```

This cuts prompt tokens by 30–50% on dense documents without information loss.

### 5. Orthogonal Axes: Table Type vs Document Type

A bank statement may contain a commercial-lines table. An invoice may contain a bank-transaction history. A payslip may contain an employment timeline.

TrustLens treats **table type and document type as orthogonal**. Every topology rule auto-applies to tables of its type, regardless of what the document claims to be. A document-type profile only adds constraints that require global-fact combinations (e.g. "opening balance matches first row").

This is a core architectural decision: **rules must not assume the document's identity constrains its internal structure.**

### 6. Versioned Regulatory Constants

Statutory rates (EPF, SOCSO, EIS, …) are stored in a YAML file keyed by `(country, effective_date)`. A reconciliation rule looks up the rate that was in force at the document's declared date — not today's rate. Adding a new regulation version never touches historical entries.

### 7. The Detective Is Not a Classifier

The reasoning VLM at the top is instructed to behave like an investigator, not a scorer:

> *"Look for small details that — combined across different sources — suggest something might be off. A single weak signal may be nothing; three weak signals pointing the same direction are worth reporting."*

Its output is a **narrative risk**, not a confidence score. Every risk cites explicit evidence IDs and/or observation IDs.

---

## Engine Architectures

Each engine follows the same internal pattern:

```
Collect → Parse/Extract → Analyze → Produce (Evidence + Context)
```

But each solves a fundamentally different forensic question.

---

### 1. Perception

**Question**: *What physically exists in this document?*

Perception converts any input (PDF, scanned PDF, image) into a unified physical representation — and, in parallel, runs a VLM to extract structured DTO IR from annotated page renders.

#### Architecture

```
Input → [Native PDF / Non-native PDF / Image] → DocumentIR
                       │
                       └── (parallel) → Rendered + Annotated pages
                                              │
                                              └── Dual-Channel VLM
                                                    │
                                        ┌───────────┴────────────┐
                                        ▼                        ▼
                               ReconciliationDTOIR      GroundingDTOIR
```

#### Highlights

- **Three input paths unified.** Native PDFs use PyMuPDF text extraction; scanned PDFs are rendered then OCR'd; standalone images go directly to OCR. All three produce the same `ObservationIR` schema.

- **Page-level native detection.** Each page is classified independently — a PDF that is 90% native text with one scanned attachment is handled correctly, not forced into a single path.

- **Table reconstruction with dual strategies.** Grid-based tables (from PyMuPDF) are reconstructed directly; bbox-only tables (from Docling) are reconstructed via **scanline projection with gap-peak detection** to infer column boundaries. Colspan detection uses character-center alignment.

- **Container marker + Y-axis fragmenting.** When Docling produces a large container box that wraps multiple parallel items, the container is not discarded. Its residual observations are split into Y-axis fragments so no content is lost.

- **Suppression of redundant regions.** After disabling Docling's table-structure module (to avoid double-processing), each table cell becomes a separate paragraph region. These are suppressed if 85%+ contained by any table bbox — the table reconstructor handles them instead.

- **Dual-channel DTO IR VLM pipeline.** Two parallel VLM channels extract:
  - **Reconciliation channel** — document type, global facts, tables.
  - **Grounding channel** — entities with structured keys.

  Both channels share a single rendering pass, a single `ObservationMapper`, and a single global VLM semaphore.

- **No `response_schema`, prompt-described schema.** Gemini's constrained decoder handles deeply-nested `list[list[...]]` poorly. TrustLens uses `response_mime_type=application/json` with a schema description in the prompt, and validates the result with Pydantic downstream. Side benefit: **35% fewer prompt tokens.**

- **Annotated images render `observation_id` directly on text.** The VLM sees "1026" printed next to a line of text and can cite it directly — no coordinate-based guessing.

- **Prompt explicitly declares table_type independent of document_type.** A long list of examples teaches the VLM that an INVOICE may contain a BANK_TRANSACTIONS table, and that skipping a table because "this document type shouldn't have it" is an extraction failure.

- **Multi-page PDF orchestration via mixed execution.** Native pages are processed serially in the main process (Docling loads its model once for the whole PDF). Non-native pages are processed in a `ProcessPool` (each worker gets an isolated Docling instance with `do_ocr=True`). The two branches run in parallel threads.

- **DTO IR is generated in parallel with DocumentIR construction.** The VLM call runs on its own thread while the CPU-bound DocumentIR is being built.

---

### 2. Metadata

**Question**: *Does the file container tell a story that contradicts the document's claims?*

Metadata inspects the file at the container level — EXIF, XMP, PDF internals, signatures, object graph — and cross-references what it finds against expected software behavior.

#### Architecture

```
ExifTool ─┐
qpdf ─────┼──▶ Collectors ──▶ MetadataContainer ──▶ Analyzers ──▶ Evidence
pikepdf ──┤                       │
PyMuPDF ──┤                       ▼
pdfsig ───┘                  ContextBuilder ──▶ MetadataContext
```

Four layers, strictly separated:

| Layer | Responsibility | Produces Evidence? |
|---|---|---|
| **Collector** | Runs external tools, captures raw output | ❌ No |
| **Parser** | Deep binary/structural analysis | ❌ No |
| **Analyzer** | Cross-source reasoning | ✅ Yes |
| **Sanitization** | Context cleaning / aggregation | ❌ No |

This strict separation means a Collector never decides anything — it just fetches. All judgment happens in Analyzers, where cross-source data is visible.

#### Highlights

- **Multi-tool triangulation.** ExifTool for metadata, qpdf for structural validity, pikepdf for object graph, PyMuPDF for fonts and layout signals, pdfsig for signature byte-range integrity, and `cryptography` for certificate date cross-validation. Each tool covers a different failure mode.

- **Fingerprint registry with ~30 producers.** Producer identification uses a YAML registry, matching against metadata *and* binary headers. Each producer has a `(document_type → risk_level)` table — "Adobe Photoshop" on a payslip is `critical`, on a resume it's `low`.

- **Binary header fingerprinting.** The second line of a PDF file (a binary comment) contains generator-specific magic bytes. TrustLens extracts and hex-encodes this for fingerprint matching — a signal that metadata alone cannot fake.

- **JPEG DQT/DHT complete parsing.** Quantization tables are decoded with full 8-bit/16-bit depth support. Huffman tables are compared byte-for-byte against ITU-T T.81 Annex K reference tables, producing `standard` / `optimized` / `mixed` classification.

- **Timeline builder preserves source labels.** Every time value — from PDF metadata, XMP, EXIF, or filesystem — is normalized to ISO-8601 but retains its original source tag. The context never collapses "which clock said this" into "the date is X".

- **Strict whitelist date parsing.** No blind guessing. If a date string matches no known format, it is dropped (and logged), not approximated. A wrong timestamp is worse than a missing one.

- **Context contains only metadata-layer-unique signals.** Font *usage* distribution is produced by the Visual engine. Font *registration-but-unused* is produced by Metadata. Image DPI is Metadata-only. This avoids duplication across Contexts.

- **Two output channels.** Evidence feeds the pipeline; `MetadataContext` feeds the Detective with a curated "case scene": identity, software provenance, timeline, XMP history, PDF integrity, revision history, active content, embedded files, object graph.

- **`makernotes_present` cross-check.** A camera-captured image with `Make`/`Model` but no MakerNotes is a strong tampering signal — TrustLens surfaces this as an observation, not a conclusion.

---

### 3. Visual

**Question**: *Was the page content rendered by a single, consistent process?*

Visual analyzes typography, character geometry, vector structure, and image composition. It works on both native PDFs (span-level extraction) and scanned/image inputs (character segmentation).

#### Architecture

```
DocumentContext ─▶ SourceTypeDetector
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   Native PDF                        Image / Scan
   (span extraction)                 (CameraDigitalClassifier
        │                             → ImageCharSegmenter)
        └────────────┬───────────────────┘
                     ▼
                 VisualIR
                     │
        ┌────────────┼────────────┬─────────────┐
        ▼            ▼            ▼             ▼
   Typography   CharSpacing   Overlap    Outlining/Vector
        │            │            │             │
        └────────────┴────────────┴─────────────┘
                     ▼
        ┌────────────┴────────────┐
        ▼                         ▼
   Evidence                  VisualContext
```

#### Highlights

- **Source-type aware, per page.** Native PDFs use PyMuPDF raw span/char extraction for maximum precision. Scanned pages are character-segmented from OCR line boxes via connected components + X-axis merge + Y-axis density protection.

- **Camera page rejection before analysis.** A `CameraDigitalClassifier` runs first and, using three orthogonal features — *solid-color ratio*, *histogram peak width*, *peak concentration* — distinguishes software-rendered pages from camera photographs. Camera pages are skipped (they cannot be visually faked).

- **Typography baseline at three independent scopes.**
  - **Global** — rare font/size/color across the whole document.
  - **Page** — rare within one page (catches "page 1 uses page 2's dominant font").
  - **Element** — rare within a semantic unit (catches inter-element style drift).

  Results from all three are merged. A span flagged at multiple scopes gets a confidence bonus.

- **Histogram weights by character count, not span count.** A long span with common styling shouldn't be treated as an outlier just because it's long.

- **Log-space MAD for rare-bar detection.** Histograms are log-transformed before MAD, allowing the "long tail" to be detected without arbitrary percentage cutoffs.

- **Dixon Q test / MAD switching.** For small-sample groups (n ≤ 10), a Dixon Q outlier test is used. For n ≥ 11, a modified Z-score against MAD. The threshold table is explicit per sample size.

- **Dynamic kerning tolerance.** Character overlap detection uses a base tolerance of 12% of the narrower character's width, relaxed to 28% or 45% when specific punctuation + right-hanging character pairs (`4,`, `r.`, `T:`) are detected. Two-dimensional vertical decoupling prevents false positives from subscripts.

- **Four overlap sub-detectors.**
  - **Occlusion** — a large opaque object covering ≥ 90% of another.
  - **Object reuse** — same vector instruction hash or image digest at distinct positions.
  - **Overlay characterization** — semi-transparent vector or image overlay.
  - **Copy-move correlation** — occlusion + reuse on the same pair → high confidence.

- **Translation-invariant hashing.** Vector drawings are hashed by their instruction sequence with coordinates normalized to the bbox origin. The same shape at a different position produces the same hash — enabling reliable reuse detection.

- **RANSAC baseline fitting with character reliability tiers.** Baseline analysis uses three tiers of characters:
  - **Reliable** — anchors for RANSAC.
  - **Uncertain** (`g j p q y`) — only downward residuals allowed.
  - **Excluded** — punctuation, ignored.

- **Header forgiveness for column alignment.** If ≥ 2 columns of the first two rows are outliers, all of them are forgiven — header rows legitimately have irregular alignment.

- **Every analyzer produces anomalies AND context.** The anomalies become Evidence. The context — per-column MAD stats, baseline slopes, duplicate groups — is passed to the Detective as raw observation.

---

### 4. Reconciliation

**Question**: *Do the numbers add up?*

Reconciliation verifies internal arithmetic, timeline consistency, statutory rates, and identifier validity. Rules are organized by **mathematical axiom**, not by document type.

#### Architecture

```
ReconciliationDTOIR
        │
        ├─▶ universal_rules()     (all docs)
        ├─▶ common_rules()        (by table type)
        └─▶ profile_rules()       (by doc type)
                │
                ▼
        RuleContext ──▶ Rules ──▶ RuleResult
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
               Evidence (FAILED)          ReconciliationContext
                                          (all statuses)
```

#### The Six Topologies

| Topology | Axiom |
|---|---|
| `state_transition` | `balance[t-1] + in[t] − out[t] = balance[t]` |
| `product_integrity` | `qty × unit_price − discount + tax = row_total` |
| `additive_partition` | `Σ(earnings) − Σ(deductions) = net_pay` |
| `temporal_interval` | `start ≤ end`; date within period |
| `statistical` | Benford's Law first-digit distribution (MAD) |
| `identifiers` | Luhn, MyKad format, reference-embedded dates |

#### Highlights

- **Table type and document type are orthogonal.** `common_rules()` auto-applies to any table of the matching type — regardless of the document. A bank statement with a commercial table gets commercial checks. An invoice with a bank-transaction table gets bank-state checks.

- **Four-state rule result: PASSED / FAILED / SKIPPED / INCOMPLETE.** This distinguishes:
  - **SKIPPED** — the rule simply doesn't apply (missing column, missing fact).
  - **INCOMPLETE** — the rule *should* have run but data is missing.

  The distinction matters: a forger's best tool is data absence. `INCOMPLETE` preserves the "we wanted to check this but couldn't" signal.

- **Evidence only from FAILED.** Every other state is recorded in the Context, letting the Detective see the baseline (how many rules passed, what was skipped, what couldn't be run) rather than a wall of anomalies.

- **Decimal everywhere.** All money math goes through `Decimal`. `float` is banned. Safe division returns `None` instead of raising. Currency conversion uses ISO 4217 normalization.

- **Statutory rates are versioned.** A YAML file maps `(country, effective_from) → rates`. The rule selects the version in force at the document's declared date. Adding 2026 EPF changes doesn't touch 2023 entries.

- **Conservative identifier checks.**
  - **Luhn** is only applied to strings that look like card numbers (13–19 pure digits). Malaysian bank accounts without Luhn are not penalized.
  - **MyKad** format is validated structurally (date, state code, serial). The check digit is *not* verified — the JPN algorithm isn't public, and guessing would create false positives.

- **Embedded-date extraction with year guards.** Reference numbers like `INV-20240615-001` have their embedded dates extracted — but only if the year is in `[2015, 2040]`. Arbitrary 8-digit numbers are not treated as dates.

- **Cascade failure aggregation.** When consecutive rows of a running-balance chain fail, they are aggregated into a single FAILED result with a summed delta — not reported row-by-row. A forger's edit at row 10 that breaks rows 10–30 produces one finding, not twenty.

- **Benford with proper Nigrini zones.** MAD-based conformity classification at three thresholds: close (< 0.006), acceptable (< 0.012), marginal (< 0.015), nonconformity (≥ 0.015). Marginal-conformity cases raise severity to warning without flagging as FAILED.

- **A `RuleContext` is immutable during a run.** Rules do not mutate shared state. `TableInstance` uses internal IDs — never the VLM-generated `table.id` — preventing forged IDs from injecting behavior.

- **`SUBTOTAL` `DISCOUNT` `TAX` `SHIPPING` composition** is checked as a single rule, but `SUBTOTAL = Σ(row_totals)` and `TAX = Σ(row_taxes)` are checked separately. A subtotal that matches neither is flagged in both directions.

---

### 5. Grounding

**Question**: *Does the outside world agree with what this document claims?*

Grounding verifies entities against authoritative sources — government registries, industry listings, domain registration, and (as a fallback) web search.

#### Architecture

```
GroundingTargets
        │
        ▼
GroundingStrategyRouter ──▶ route (per entity type)
        │
        ├─▶ Deterministic backends   (SSM / BNM / WHOIS / Whitelist)
        ├─▶ Enterprise connector     (placeholder; DB)
        └─▶ Web search               (DuckDuckGo + LLM summarizer)
                │
                ▼
        Three parallel streams
                │
                ▼
        GroundingContext (no Evidence)
```

#### Five-State Outcome

| Outcome | Meaning |
|---|---|
| `EXACT_MATCH` | An authoritative source confirms the exact value. |
| `FUZZY_MATCH` | Related values found, but no single source confirms exact. |
| `CONFLICT_FOUND` | An authoritative source confirms a *different* value for the same identifier. |
| `NOT_FOUND` | Authoritative sources were searched and none returned a match. |
| `UNVERIFIABLE` | No authoritative source exists or is reachable. |

The system never conflates `NOT_FOUND` with "fake". A `NOT_FOUND` result means "we looked and didn't find", which is observation, not conclusion.

#### Highlights

- **Three source streams, no shared base class.** Web / enterprise / deterministic results are structurally different — trying to unify them into a single type would collapse semantics ("authoritative registry entry" is not "web search snippet"). They stay separate; the Detective reads each.

- **Deterministic routing with YAML override.** Entity type maps to a route deterministically. An optional YAML file can override per-entity-type routes without code changes.

- **Three source streams run in parallel.** The four routes (deterministic per backend, enterprise, web, unverifiable) execute concurrently in a `ThreadPoolExecutor`.

- **Funneled web fallback.** Every deterministic backend returning `NOT_FOUND` or `UNVERIFIABLE` is collected and sent to a *single* web-search call at the end. This reduces LLM summarizer invocations from N (one per target) to 1 (one for the whole batch).

- **LLM summarizer with strict count enforcement.** One LLM call summarizes N queries. If the LLM returns fewer items than expected, the count mismatch is logged and the missing entries are marked as errors — no silent truncation.

- **DuckDuckGo as the primary web backend.** Free, no API key, works out of the box. Tavily support is retained in code for future deployment scenarios.

- **Gemini grounding with Google Search** as an alternative path — used for cases where the LLM should reason directly over search results rather than receive pre-fetched snippets.

- **No Evidence output, by design.** Grounding produces observations ("BNM's consumer-alert list contains this name", "RDAP shows domain registered 3 days ago"). Whether that observation implies fraud is a decision for the Detective.

- **BNM double-layer: consumer alerts + FSP directory.** The Bank Negara Malaysia backend queries both the public consumer-alert blacklist and the licensed financial-services-provider whitelist. A match in either is authoritative.

- **RDAP instead of WHOIS.** Structured JSON, no rate-limit hell, IANA-standardized. The backend extracts registration/expiration dates and registrar identity.

- **Whitelist name normalization.** Malaysian company names have systematic suffix variations (`Sdn Bhd`, `Sdn. Bhd.`, `Bhd`, `Berhad`, …). A normalizer strips these and produces a canonical key, allowing exact-match lookup against registry snapshots.

---

### 6. Semantic

**Question**: *Does the document mean what it says — and does it say what it means?*

Semantic analyzes the meaning of the document's text: missing information, internal contradictions, one-sided clauses, and material ambiguities.

#### Architecture

```
DocumentIR.elements
        │
        ▼
   serialize_elements()  →  "[paragraph] obs=1023-1038 | <text>"
        │
        ▼
   Gemini (single chunk by default)
        │
        ▼
   JSON evidence extraction
        │
        ▼
   Evidence list  (SEMANTIC_GAP / CONTRADICTION / UNFAIR_CLAUSE / AMBIGUITY)
```

#### Four Evidence Categories

| Category | What it captures |
|---|---|
| `SEMANTIC_GAP` | Required information is missing, undefined, or referenced but unavailable. |
| `SEMANTIC_CONTRADICTION` | Two parts of the document cannot be reconciled. |
| `SEMANTIC_UNFAIR_CLAUSE` | One-sided or unfair terms, primarily in contracts. |
| `SEMANTIC_AMBIGUITY` | Language whose interpretation is materially uncertain. |

#### Highlights

- **The prompt is a method, not a taxonomy.** The system prompt explicitly says:

  > *"The categories are reporting categories, NOT an exhaustive model of semantic reasoning. Do not restrict your reasoning to the examples listed under each category."*

  This prevents the LLM from treating the four categories as a checklist to fill in.

- **"Do not prove fraud."** The prompt explicitly tells the LLM it is producing forensic evidence, not a verdict. It is told that a downstream Detective will combine its output with other engines and produce the final assessment.

- **Every finding must quote exact text.** No paraphrasing. If a problem spans elements, all fragments are listed. `observation_ids` must reference the elements containing the quoted fragments — an ungrounded finding is invalid.

- **Justification in the source language.** The prompt requires the justification to be written in the same language as the quoted text. This prevents language-mismatched reasoning about documents the LLM may have been trained primarily in English.

- **Dynamic confidence based on evidence strength.**
  - Base 0.7 for a stated quotation + justification.
  - +0.05 if the justification is detailed (>100 chars).
  - +0.10 if there are ≥ 2 supporting quotations.
  - +0.10 if external sources are cited.

  Capped at 0.9 — semantic judgments should never claim certainty.

- **Optional web search, budget 1.** If the LLM needs to check a specific legal or regulatory reference, it may use at most one web search. Web results support the justification but never replace document evidence.

- **Default single chunk.** 99% of documents fit in one LLM call. The chunking path exists for very long documents but is off by default.

- **Web search tool is off by default.** Most semantic problems don't require external verification. Enabling web search is opt-in.

- **Output is Evidence only, no Context.** This breaks the dual-channel pattern deliberately: semantic analysis has no "neutral observation" mode — its only output is a claimed problem. The Detective receives semantic findings as Evidence, not as Context.

---

### 7. Detective

**Question**: *Given all of this, what happened?*

The Detective receives a **CaseFile** — a curated dossier of everything the five engines found — along with the annotated page images. It reasons across all sources, produces a narrative risk report, and cites explicit evidence IDs for every claim.

#### Architecture

```
DocumentIR ──┐
Evidences ───┤
Metadata ────┤
Visual ──────┼──▶ CaseFileBuilder ──▶ CaseFile
Recon ───────┤                            │
Grounding ───┘                            │
                                          ▼
                                    CaseFileRenderer
                                          │
                                          ▼
                                    Markdown prompt
                                          │
                                          ▼
                            DetectiveVLMClient (with annotated images)
                                          │
                                          ▼
                                     parse_report
                                          │
                                          ▼
                                     DetectiveReport
```

#### CaseFile Structure

The CaseFile is **not a raw dump**. It is a curated case dossier:

| Section | Treatment |
|---|---|
| **DocumentIR elements** | Full reading-order projection + `observation_text_map` (obs_id → text + bbox) |
| **Metadata Context** | Full dict (user requirement) |
| **Visual Context** | Full dict |
| **Reconciliation Context** | Structured: FAILED folded into Evidence; INCOMPLETE preserved verbatim; PASSED/SKIPPED kept as counts |
| **Grounding Context** | Full dict, with raw search responses and matched records stripped |
| **Evidence List** | Indexed (`E001`, `E002`, …), grouped (`G001`, …), sorted |

#### Highlights

- **Reconciliation is handled structurally, not textually.** The reconciliation engine produces hundreds of rule results, most of which pass. Dumping all of them would blow the prompt. Instead:
  - **FAILED** results are already in the Evidence list — not repeated.
  - **INCOMPLETE** results preserve their full description and reason, because "we couldn't check this" is materially important.
  - **PASSED / SKIPPED** results are reduced to counts per rule name.

- **Evidence is indexed and grouped.** Every Evidence becomes `E###`. Evidences sharing a `(type, table_id)` pair get a shared `G###`. This lets the Detective cite "E001 through E005, all in group G002" instead of restating each.

- **Sorting creates adjacency.** Evidence is sorted by `(source_engine, page, bbox.y0)`. Related evidence ends up physically adjacent in the prompt — the Detective's attention mechanism can pick up on local clusters.

- **`observation_text_map` grounds every ID.** When the Detective reads `obs=1026-1036`, it looks up each ID in the map and finds both the exact text and the bounding box. This is what makes citation-style reasoning possible.

- **The prompt teaches *how to think*, not *what to classify*.**

  > *"You are NOT a classifier. You do NOT need to prove fraud. You ARE an investigator. Look for small details that — combined across different sources — suggest something might be off."*

  The prompt contains worked examples ("a timeline contradiction between different time sources", "a rare font appearing in exactly the wrong place") but presents them as *patterns*, not categories.

- **`risk` is a narrative, not a two-part structure.** The schema rejects "finding: X, justification: Y" in favor of "risk: <coherent 2–5 sentence narrative>". This forces the LLM to write causally rather than atomically.

- **Confidence is defined as investigator certainty, not signal strength.**

  > *"high — multiple independent signals converge, OR one decisive signal"*
  > *"medium — one or two signals with reasonable interpretation"*
  > *"low — a pattern warrants a closer look but could be benign"*

- **Citation rules are explicit.**
  - `evidence_ids` for engine-produced Evidence.
  - `observation_ids` for the Detective's own reading.
  - Both may be used together.
  - Inventing IDs is forbidden.

- **Annotated images ship with the prompt.** The Detective receives the page images with observation IDs drawn next to text — so it can visually verify a citation by looking at the render.

- **An empty risks array is a valid output.** If the Detective finds nothing material, it says so and explains why in the summary. This makes a "clean" verdict informative rather than just silent.

---

## Walkthrough

Consider a forged payslip that passes metadata and typography checks.

```
Perception
  ├─ Native PDF path
  ├─ observations: 47 lines, 2 tables
  └─ DTO IR: PAYSLIP, PAYROLL_COMPONENTS + EMPLOYMENT tables

Metadata (parallel)
  ├─ Producer: "Microsoft Word"
  ├─ XMP history: single step
  └─ → no Evidence

Visual (parallel)
  ├─ TypographyAnalyzer: all spans in dominant triple
  └─ → no Evidence

Reconciliation (parallel)
  ├─ Σ(earnings) = 8400.00, GROSS_PAY = 8400.00       ✓
  ├─ Σ(deductions) = 1120.00, EMPLOYEE_DEDUCTION_TOTAL = 1120.00  ✓
  ├─ GROSS_PAY − DEDUCTIONS = 7280.00, NET_PAY = 7280.00  ✓
  ├─ EPF rate 11.5% ≠ statutory 11% (tolerance 0.5%)
  └─ → Evidence: RECONCILIATION_STATUTORY_RATE_MISMATCH

Grounding (parallel)
  ├─ Employee name → unverifiable (by design)
  └─ → no Evidence

Semantic (parallel)
  ├─ Reads all elements
  └─ → no Evidence

Detective
  ├─ Reviews all evidence + contexts
  ├─ Notices: EPF rate is 0.5pp above statutory; total arithmetic still passes
  └─ → RiskItem {
       risk: "EPF employee rate of 11.5% does not match any statutory schedule.
              While the arithmetic across gross, deductions, and net remains
              internally consistent, the rate itself is not achievable under
              Malaysian EPF rules for any salary band.",
       evidence_ids: ["E007"],
       observation_ids: ["2043-2044"],
       confidence: "high"
     }
```

The final output is not "0.87 risk". It is a case file.

---

## Project Structure

```
engine/
├── app/
│   ├── core/                        # Shared models: DocumentContext, DTO IR, Evidence
│   ├── orchestration/               # ForensicPipeline + PipelineConfig
│   ├── perception/                  # Physical extraction + DTO IR
│   │   ├── extractors/              # PyMuPDF, RapidOCR, Docling
│   │   ├── builders/                # Region assignment, table reconstruction
│   │   ├── dto_ir/                  # VLM dual-channel pipeline
│   │   └── orchestration/           # Multi-page PDF orchestrator
│   └── forensics/
│       ├── metadata/                # File container forensics
│       ├── visual/                  # Page content forensics
│       ├── reconciliation/          # Internal arithmetic
│       ├── grounding/               # External world verification
│       └── semantic/                # Meaning-level analysis
├── detective/                       # CaseFile + VLM reasoning
└── data/
    ├── whitelist/                   # Bursa, MCMC, NPRA JSON snapshots
    ├── fingerprints.yaml            # Producer fingerprint registry
    └── statutory_rates.yaml         # Versioned regulatory constants
```

---

## Tech Stack

| Component | Purpose |
|---|---|
| **PyMuPDF** | PDF text, image, drawing, annotation extraction |
| **pikepdf** | Object graph traversal, embedded files |
| **Docling** | Layout analysis, semantic region detection |
| **RapidOCR** | OCR for scanned documents |
| **ExifTool / qpdf / pdfsig** | External CLI tools for metadata forensics |
| **Gemini (Vertex AI)** | VLM for DTO IR extraction, web summarization, semantic analysis, and final reasoning |
| **DuckDuckGo / Tavily** | Web search for grounding fallback |
| **scikit-learn** | RANSAC for baseline fitting |
| **Pydantic v2** | All models, strict validation, JSON serialization |
| **Decimal** | All money arithmetic (no floats) |

---

## Status & Roadmap

TrustLens v2 is an active research system.

| Area | Status |
|---|---|
| Perception (all three input paths) | ✅ Implemented |
| Metadata (ExifTool / qpdf / pikepdf / PyMuPDF / pdfsig) | ✅ Implemented |
| Visual (native + scanned) | ✅ Implemented |
| Reconciliation (six topologies) | ✅ Implemented |
| Grounding (BNM / WHOIS / Whitelist / Web) | ✅ Implemented |
| Grounding (enterprise DB connector) | 🚧 Placeholder |
| Semantic | ✅ Implemented |
| Detective | ✅ Implemented |
| End-to-end orchestration | ✅ Implemented |

Documented improvement areas:

- Cross-layer coordinate system labeling (PDF points vs image pixels).
- Evidence chaining across engines — a shared "case graph" where pieces of evidence can reference each other.
- Expanded statutory rate tables (currently Malaysia only).
- Additional authoritative registries (universities, professional bodies, law firms).
- Enterprise DB connectors for real-world integration.

---

## Philosophy

TrustLens is built on the belief that **forensic analysis of documents is a reasoning problem, not a classification problem.**

The system does not attempt to output a single score. It gathers evidence from multiple independent physical and semantic layers, presents it in a form that a reasoning model can consume, and asks that model to build a coherent case.

The final output is not "0.87 risk". It is:

> *"Here is what we noticed, here is why it matters, and here is where in the document you can find it."*

That is what a detective would say.

---

*TrustLens v2 — because forging a document requires failing many checks at once, and matching it requires noticing many things at once.*