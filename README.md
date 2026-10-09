# TrustLens v2

**Document Forensics Through Multi-Engine Evidence Fusion**

TrustLens is a document forensics system that determines whether a business or legal document (invoice, payslip, bank statement, contract, certificate, resume, etc.) has been forged, altered, or misrepresented. It does so by running five independent forensic engines across a shared physical representation of the document, then fusing their outputs through a reasoning VLM that plays the role of a detective.

The design principle is simple:

> **No single engine decides. Evidence accumulates. A detective reasons.**

---

## Why TrustLens

Traditional document verification relies on a single signal — a metadata check, a visual inspection, or an arithmetic audit. Real forgery rarely fails on just one axis. A forged payslip might have:

- clean metadata (produced by a common PDF library),
- clean typography (rendered in a single pass),
- **but** a running balance that doesn't close,
- **and** a net pay that doesn't equal gross minus deductions,
- **and** an EPF rate that matches no statutory schedule.

No single engine would flag this. TrustLens would.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        ForensicPipeline                            │
│                    (top-level orchestrator)                        │
└────────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌──────────────────┐      ┌───────────────┐
│  Perception   │      │  5 Engines        │      │   Detective   │
│               │      │  (parallel)       │      │               │
│  • Documents  │      │                   │      │  CaseFile     │
│  • DTO IR     │      │  1. Metadata      │      │  + annotated  │
│  • Annotated  │      │  2. Visual        │      │    images     │
│    images     │      │  3. Reconciliation│      │  → Detective  │
│               │      │  4. Grounding     │      │    Report     │
│               │      │  5. Semantic      │      │               │
└───────────────┘      └──────────────────┘      └───────────────┘
        │                        │                        ▲
        │                        ▼                        │
        │               ┌──────────────────┐              │
        └──────────────▶│  Evidence List   │──────────────┘
                        │  + Contexts      │
                        └──────────────────┘
```

### Layer Descriptions

| Layer | Responsibility | Output |
|---|---|---|
| **Perception** | Physical extraction: text, tables, images, and a structured DTO IR | `DocumentIR`, `ReconciliationDTOIR`, `GroundingDTOIR`, annotated pages |
| **Metadata** | File container layer: EXIF, XMP, PDF structure, signatures, object graph | `MetadataContext` + Evidence |
| **Visual** | Page content layer: typography, character geometry, overlaps, vector spoofing, camera/digital classification | `VisualContext` + Evidence |
| **Reconciliation** | Internal arithmetic: conservation laws, statutory rates, timelines, identifiers | `ReconciliationContext` + Evidence |
| **Grounding** | External world: authoritative registries, enterprise DB, web search | `GroundingContext` (no Evidence) |
| **Semantic** | Meaning-level: gaps, contradictions, unfair clauses, ambiguities | Evidence |
| **Detective** | Cross-source reasoning: fuses everything into a risk report | `DetectiveReport` |

---

## Design Principles

### 1. Evidence and Context are two separate channels

Every engine produces two outputs:

- **Evidence** — machine-checkable facts, e.g. `RECONCILIATION_RUNNING_BALANCE_MISMATCH`.
- **Context** — structured observations that require reasoning, e.g. an unusual XMP history chain.

The Detective consumes both. This avoids two failure modes:
- boiling everything down to "risk scores" (loses nuance),
- drowning the LLM in raw data (loses signal).

### 2. Observation IDs are the universal anchor

Every physical line of text is assigned a globally-unique ID:

```
observation_id = page * 1000 + local_index
```

This single ID flows through every layer — from OCR, through DTO IR extraction, into every piece of Evidence. When the Detective cites a finding, it cites an `observation_id`. When we re-render an annotated image, the same ID is drawn on top of the source text. The result: **every claim is traceable back to a specific location in the original document.**

### 3. Compressed ranges everywhere

`[1026, 1027, 1028, 1045, 1046]` → `["1026-1028", "1045-1046"]`

This cuts prompt tokens by 30-50% on dense documents without losing information.

### 4. Graceful degradation, never silent failure

Every engine runs inside its own try/except. If Metadata fails, Visual still runs. If the Visual VLM call times out, other engines are unaffected. Failures are recorded in a top-level `errors` list, and the Detective is informed about which engines actually ran.

### 5. Mathematical axioms, not heuristics

Reconciliation rules are organized by **topological class**, not by document type:

| Topology | Axiom |
|---|---|
| `state_transition` | `balance[t-1] + in[t] - out[t] = balance[t]` |
| `product_integrity` | `qty × unit_price - discount + tax = row_total` |
| `additive_partition` | `Σ(earnings) - Σ(deductions) = net_pay` |
| `temporal_interval` | `start ≤ end` |
| `statistical` | Benford's Law (MAD conformity) |

A bank statement containing a commercial-lines table is still subject to product integrity rules. **Table type and document type are orthogonal.**

### 6. The Detective is not a classifier

The VLM at the top is instructed to behave like an investigator, not a scorer:

> *"Look for small details that — combined across different sources — suggest something might be off. A single weak signal may be nothing; three weak signals pointing the same direction are worth reporting."*

Its output is a **narrative risk**, not a confidence score. Every risk cites explicit evidence IDs and/or observation IDs.

---

## Engine Deep-Dives

### Perception

Renders the physical document into a unified `DocumentIR`:

- **Native PDF** → PyMuPDF text + Docling layout
- **Scanned PDF** → rendered to image, RapidOCR + Docling OCR
- **Pure image** → RapidOCR + Docling OCR

Table reconstruction handles both PyMuPDF grid-based tables and Docling bbox-only tables via scanline projection with gap-peak detection.

Alongside, a **dual-channel VLM pipeline** (Gemini) extracts structured data:
- Channel 1 (Reconciliation): document type, global facts, tables
- Channel 2 (Grounding): entities with structured keys

The two channels run in parallel with a shared rendered annotation set, and outputs are validated against the physical observation layer.

### Metadata

| Collector | Purpose |
|---|---|
| ExifTool | EXIF / XMP / PDF metadata, complete raw dump |
| qpdf | Structural validity, revisions, encryption |

| Analyzer | Detects |
|---|---|
| XMPAnalyzer | History chains, CreatorTool/Producer mismatch |
| ConsistencyAnalyzer | Timeline contradictions, encoding anomalies, image structural fingerprints |
| FingerprintAnalyzer | Producer fingerprint matching + Creator→Producer category chain |
| SignatureAnalyzer | Certificate expiry, timestamp mismatches, multi-signature inconsistency |

A YAML fingerprint registry with ~30 producers maps `(producer, document_type) → risk_level`, matching against both metadata and binary headers.

### Visual

Source-type aware: native PDFs use span/char-level extraction; scanned pages use character segmentation from OCR lines.

| Analyzer | What it finds |
|---|---|
| TypographyAnalyzer | Style outliers at global / page / element scope |
| CharSpacingAnalyzer | Character overlaps (dynamic kerning tolerance), digit-width outliers (Dixon Q / MAD) |
| OverlapAnalyzer | Occlusion, object reuse, overlay characterization, copy-move correlation |
| OutliningAnalyzer | Partial vector outlining of text |
| VectorSpoofingAnalyzer | Micro vector strokes over characters |
| ImageBaselineAnalyzer | RANSAC baseline fitting with three-tier character reliability |
| ImageAlignmentAnalyzer | Table column alignment outliers with header forgiveness |

A `CameraDigitalClassifier` filters out camera-captured pages before analysis — histogram peak width and solid-color ratio distinguish software-rendered images from photographs.

### Reconciliation

Document-type rules are split into three tiers:

1. **Universal rules** — apply to every document (dates in future, currency consistency, identifier checks).
2. **Common topology rules** — auto-applied by *table type* (state transition, product integrity, additive partition, statistical).
3. **Profile rules** — document-specific (e.g. bank period containment, payslip period length).

Statutory constants (EPF, SOCSO, EIS rates) are versioned by `(country, effective_date)` in YAML, allowing rules to use the correct rate for the document's date.

### Grounding

A deterministic router dispatches each target to one of four paths:

| Entity Type | Route |
|---|---|
| BANK | BNM (consumer alerts + FSP directory) |
| WEBSITE | RDAP / WHOIS |
| ORGANIZATION / VENDOR | Local whitelist (Bursa, MCMC, NPRA) |
| ACCOUNT / TRANSACTION / INVOICE / ... | Enterprise DB (placeholder) |
| PERSON / CUSTOMER / PRODUCT / ADDRESS | Unverifiable (by design) |
| OTHER | Web search fallback |

Web search uses **DuckDuckGo** (free, no key) with Gemini summarization per query. Tavily support is retained for future deployment.

Any deterministic backend returning `NOT_FOUND` or `UNVERIFIABLE` is batched and sent to a single web search call — reducing LLM summarizer invocations from N to 1.

### Semantic

Serializes `DocumentIR.elements` into a line-delimited format and asks the VLM to identify four categories of semantic problems:

- `SEMANTIC_GAP` — missing or incomplete information
- `SEMANTIC_CONTRADICTION` — internal conflicts
- `SEMANTIC_UNFAIR_CLAUSE` — one-sided terms
- `SEMANTIC_AMBIGUITY` — materially unclear language

Each finding includes exact quoted fragments, `observation_id` references, a justification in the source language, and optionally external sources.

### Detective

The Detective receives a **CaseFile** — a curated dossier containing:

- Annotated page images (text with observation IDs drawn on top)
- Document elements in reading order, with `observation_text_map` for reference
- Metadata / Visual / Grounding contexts (full)
- Reconciliation projection (FAILED folded into Evidence; INCOMPLETE preserved; PASSED/SKIPPED as counts)
- An indexed evidence list (`E001`, `E002`, …) with group IDs (`G001`)

It outputs a JSON report with a summary, an overall risk level, and per-risk narratives citing evidence/observation IDs.

---

## Data Flow Example

Consider a forged payslip that passes metadata and typography checks:

```
Perception
  ├─ Native PDF path
  ├─ observations: 47 lines, 3 tables
  └─ DTO IR: PAYSLIP, 2 tables (PAYROLL_COMPONENTS, EMPLOYMENT)

Metadata (parallel)
  ├─ Producer: "Microsoft Word"
  ├─ XMP history: single step
  └─ → no Evidence

Visual (parallel)
  ├─ TypographyAnalyzer: all spans in dominant triple
  └─ → no Evidence

Reconciliation (parallel)
  ├─ Σ(earnings) = 8400.00, GROSS_PAY = 8400.00 ✓
  ├─ Σ(deductions) = 1120.00, EMPLOYEE_DEDUCTION_TOTAL = 1120.00 ✓
  ├─ GROSS_PAY - DEDUCTIONS = 7280.00, NET_PAY = 7280.00 ✓
  ├─ EPF rate 11.5% ≠ statutory 11% (tolerance 0.5%)
  └─ → Evidence: RECONCILIATION_STATUTORY_RATE_MISMATCH

Grounding (parallel)
  ├─ EMPLOYEE target → unverifiable (by design)
  └─ → no Evidence

Semantic (parallel)
  ├─ Reads all elements
  └─ → no Evidence

Detective
  ├─ Reviews all evidence + contexts
  ├─ Notices: "EPF rate is 0.5pp above statutory; total arithmetic still passes"
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

---

## Status

TrustLens v2 is an active research system. Current state:

- **Perception, Metadata, Visual, Reconciliation, Semantic, Detective** — implemented and integrated
- **Grounding** — implemented for BNM / WHOIS / Whitelist / Web; enterprise DB connector is a placeholder
- **Tavily integration** — retained in code, DuckDuckGo used in development

Documented improvement areas include: cross-layer coordinate system labeling, embedding-based evidence chaining, and expanded statutory rate tables (currently Malaysia only).

---

## Philosophy

TrustLens is built on the belief that **forensic analysis of documents is a reasoning problem, not a classification problem**. The system does not attempt to output a single score. It gathers evidence from multiple independent physical and semantic layers, presents it in a form that a reasoning model can consume, and asks that model to build a coherent case.

The final output is not "0.87 risk". It is:

> *"Here is what we noticed, here is why it matters, and here is where in the document you can find it."*

That is what a detective would say.