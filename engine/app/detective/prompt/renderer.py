"""CaseFileRenderer — CaseFile → Markdown prompt。"""
from __future__ import annotations

import json
from typing import Any

from ..models.case_file import CaseFile
from .templates import (
    EVIDENCE_FOOTER,
    SECTION_DOCUMENT,
    SECTION_EVIDENCE,
    SECTION_GROUNDING,
    SECTION_METADATA,
    SECTION_RECONCILIATION,
    SECTION_VISUAL,
)


def _j(obj: Any, indent: int = 2, max_len: int = 0) -> str:
    """json dumps + 可选截断。"""
    s = json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
    if max_len and len(s) > max_len:
        return s[:max_len] + f"\n  ... (truncated, total {len(s)} chars)"
    return s


class CaseFileRenderer:

    def render(self, case_file: CaseFile) -> str:
        parts: list[str] = []
        parts.append(self._render_header(case_file))
        parts.append(self._render_document(case_file))
        if case_file.metadata:
            parts.append(self._render_metadata(case_file))
        if case_file.visual:
            parts.append(self._render_visual(case_file))
        if case_file.reconciliation:
            parts.append(self._render_reconciliation(case_file))
        if case_file.grounding:
            parts.append(self._render_grounding(case_file))
        parts.append(self._render_evidence(case_file))
        parts.append(EVIDENCE_FOOTER)
        return "\n\n".join(parts)

    # ---------- header ----------

    def _render_header(self, cf: CaseFile) -> str:
        return "\n".join([
            "# CASE FILE",
            f"- case_id: {cf.case_id}",
            f"- file_name: {cf.file_name}",
            f"- document_type: {cf.document_type}",
            f"- page_count: {cf.page_count}",
            f"- evidence_count: {len(cf.evidences)}",
            f"- annotated_images: {len(cf.annotated_images)} page(s)",
        ])

    # ---------- document ----------

    def _render_document(self, cf: CaseFile) -> str:
        lines = [SECTION_DOCUMENT, ""]
        lines.append("### Elements (reading order)")
        for elem in cf.elements:
            type_str = elem.element_type or "unknown"
            obs_str = ", ".join(elem.observation_ids) if elem.observation_ids else ""
            header = f"[{type_str}] obs={obs_str}" if obs_str else f"[{type_str}]"

            if elem.text:
                lines.append(f"{header} | {elem.text.strip()}")
            elif elem.table_text:
                lines.append(f"{header} | table:")
                for row in elem.table_text.splitlines():
                    lines.append(f"    {row}")
            else:
                lines.append(header)

        # 只列出 elements 引用到的 obs
        referenced: set[str] = set()
        for elem in cf.elements:
            for r in elem.observation_ids:
                if "-" in r:
                    a, b = r.split("-", 1)
                    try:
                        for i in range(int(a), int(b) + 1):
                            referenced.add(str(i))
                    except ValueError:
                        pass
                else:
                    referenced.add(r)

        if referenced:
            lines.append("")
            lines.append("### Observation Text Map")
            for oid in sorted(referenced, key=lambda x: int(x) if x.isdigit() else 0):
                txt = cf.observation_text_map.get(oid)
                if txt:
                    lines.append(f"  {oid}: {txt[:200]}")

        return "\n".join(lines)

    # ---------- metadata ----------

    def _render_metadata(self, cf: CaseFile) -> str:
        m = cf.metadata or {}
        lines = [SECTION_METADATA, ""]

        if m.get("metadata_identity"):
            lines.append("### Identity")
            lines.append(_j(m["metadata_identity"]))

        if m.get("software_provenance"):
            lines.append("### Software Provenance")
            for item in m["software_provenance"]:
                lines.append(f"- {item.get('source', '?')}: {item.get('value', '?')}")

        if m.get("timeline"):
            lines.append("### Timeline")
            for item in m["timeline"]:
                lines.append(
                    f"- {item.get('time', '?')}  (source: {item.get('source', '?')})"
                )

        if m.get("xmp_history"):
            lines.append("### XMP History")
            for item in m["xmp_history"]:
                lines.append(
                    f"- {item.get('action', '?')} by {item.get('software_agent', '?')} "
                    f"at {item.get('when', '?')}"
                )

        if m.get("document_lineage"):
            lines.append("### Document Lineage")
            lines.append(_j(m["document_lineage"]))

        if m.get("image_metadata"):
            lines.append("### Image Metadata")
            lines.append(_j(m["image_metadata"]))

        if m.get("pdf_integrity"):
            lines.append("### PDF Integrity")
            lines.append(_j(m["pdf_integrity"]))

        if m.get("revision_history"):
            lines.append("### Revision History")
            lines.append(_j(m["revision_history"]))

        ac = m.get("active_content") or {}
        if ac.get("javascript") or ac.get("open_action") or ac.get("launch_action"):
            lines.append("### Active Content")
            lines.append(_j(ac))

        if m.get("embedded_files"):
            lines.append("### Embedded Files")
            for ef in m["embedded_files"]:
                lines.append(
                    f"- {ef.get('name', '?')} "
                    f"({ef.get('size_bytes', '?')} bytes)"
                )

        if m.get("anomalous_regions"):
            lines.append("### Anomalous Regions (metadata layer)")
            regions = m["anomalous_regions"]
            for r in regions[:40]:
                lines.append(
                    f"- p{r.get('page')} [{r.get('type')}]: "
                    f"{str(r.get('reason', ''))[:150]}"
                )
            if len(regions) > 40:
                lines.append(f"  ... ({len(regions) - 40} more)")

        if m.get("replacement_chars"):
            lines.append("### Replacement Characters (U+FFFD)")
            for r in m["replacement_chars"][:10]:
                lines.append(f"- p{r.get('page')}: {str(r.get('text', ''))[:80]}")
            if len(m["replacement_chars"]) > 10:
                lines.append(f"  ... ({len(m['replacement_chars']) - 10} more)")

        if m.get("image_dpi"):
            lines.append("### Image DPI")
            for p, dpi in m["image_dpi"].items():
                lines.append(f"- p{p}: {dpi}")

        if m.get("image_structural_fingerprint"):
            lines.append("### Image Structural Fingerprint")
            lines.append(_j(m["image_structural_fingerprint"], max_len=3000))

        if m.get("image_observations"):
            lines.append("### Image Observations")
            for o in m["image_observations"]:
                lines.append(f"- {o}")

        return "\n".join(lines)

    # ---------- visual ----------

    def _render_visual(self, cf: CaseFile) -> str:
        v = cf.visual or {}
        lines = [SECTION_VISUAL, ""]

        if v.get("source"):
            lines.append("### Source")
            lines.append(_j(v["source"]))

        if v.get("page_summaries"):
            lines.append("### Page Summaries")
            for p in v["page_summaries"]:
                lines.append(
                    f"- p{p.get('page')}: "
                    f"{p.get('span_count', 0)} spans, "
                    f"{p.get('drawing_count', 0)} drawings, "
                    f"dominant={p.get('dominant_font')} @ {p.get('dominant_font_size')}"
                )

        if v.get("global_style_profile"):
            lines.append("### Global Style Profile")
            lines.append(_j(v["global_style_profile"]))

        if v.get("analyzer_contexts"):
            lines.append("### Analyzer Contexts")
            for name, ctx in v["analyzer_contexts"].items():
                lines.append(f"#### {name}")
                lines.append(_j(ctx, max_len=6000))

        if v.get("metadata"):
            lines.append("### Metadata")
            lines.append(_j(v["metadata"]))

        return "\n".join(lines)

    # ---------- reconciliation ----------

    def _render_reconciliation(self, cf: CaseFile) -> str:
        r = cf.reconciliation
        if r is None:
            return ""
        lines = [SECTION_RECONCILIATION, ""]

        lines.append(f"- document_type: {r.document_type}")
        lines.append(f"- document_id: {r.document_id}")
        if r.table_type:
            lines.append(f"- table_type: {r.table_type}")

        if r.normalized_global_facts:
            lines.append("### Global Facts (normalized)")
            for gf in r.normalized_global_facts:
                cur = f" {gf.get('currency')}" if gf.get("currency") else ""
                lines.append(
                    f"- {gf.get('role', '?')}: {gf.get('value_normalized', '?')}{cur}"
                )

        if r.table_summaries:
            lines.append("### Tables")
            for t in r.table_summaries:
                lines.append(
                    f"- {t.get('internal_id')} ({t.get('table_type')}) "
                    f"p{t.get('page')}: {t.get('row_count')} rows, "
                    f"cols={t.get('columns')}"
                )

        if r.incomplete_computations:
            lines.append("### Incomplete Computations (data missing)")
            for c in r.incomplete_computations:
                lines.append(
                    f"- [{c.rule_name}] {c.description} "
                    f"(reason: {c.unverified_reason or 'n/a'}"
                    + (f", row {c.row_index}" if c.row_index is not None else "")
                    + ")"
                )

        lines.append("### Passed / Skipped Counts")
        lines.append(f"- passed: {r.passed_skipped_counts.passed}")
        lines.append(f"- skipped: {r.passed_skipped_counts.skipped}")

        if r.unverified_fields:
            lines.append("### Unverified Fields")
            for u in r.unverified_fields:
                lines.append(f"- {u.get('field')}: {u.get('reason')}")

        if r.data_quality_issues:
            lines.append("### Data Quality Issues")
            for d in r.data_quality_issues:
                lines.append(f"- {d.get('issue')}: {d.get('detail')}")

        lines.append("### Summary")
        lines.append(_j(r.summary))

        return "\n".join(lines)

    # ---------- grounding ----------

    def _render_grounding(self, cf: CaseFile) -> str:
        g = cf.grounding or {}
        lines = [SECTION_GROUNDING, ""]

        if g.get("summary"):
            lines.append("### Summary")
            lines.append(_j(g["summary"]))

        if g.get("deterministic_results"):
            lines.append("### Deterministic (authoritative source)")
            for r in g["deterministic_results"]:
                lines.append(
                    f"- [{r.get('backend_name', '?')}] "
                    f"'{r.get('query_value', '?')}' → {r.get('outcome')} "
                    f"(conf={r.get('confidence')})"
                )
                for s in (r.get("sources") or []):
                    lines.append(f"    source: {s.get('title')} <{s.get('url')}>")

        if g.get("enterprise_results"):
            lines.append("### Enterprise (internal DB)")
            for r in g["enterprise_results"]:
                lines.append(
                    f"- '{r.get('entity_type', '?')}' → {r.get('outcome')} "
                    f"(source={r.get('source')})"
                )

        if g.get("web_results"):
            lines.append("### Web Search")
            for r in g["web_results"]:
                lines.append(
                    f"- '{r.get('query_value', '?')}' → {r.get('outcome')} "
                    f"(conf={r.get('confidence')})"
                )
                if r.get("resolved_value"):
                    lines.append(f"    summary: {r['resolved_value']}")
                for s in (r.get("sources") or []):
                    lines.append(f"    source: {s.get('title')} <{s.get('url')}>")

        return "\n".join(lines)

    # ---------- evidence ----------

    def _render_evidence(self, cf: CaseFile) -> str:
        lines = [SECTION_EVIDENCE, ""]
        if not cf.evidences:
            lines.append("(no evidence)")
            return "\n".join(lines)

        for ev in cf.evidences:
            header = f"[{ev.evidence_id}]"
            if ev.group_id:
                header += f" (group {ev.group_id})"
            header += f" {ev.type}  ({ev.source_module}, conf={ev.confidence:.2f})"
            lines.append(header)

            if ev.description:
                lines.append(f"  description: {ev.description}")

            val_json = json.dumps(ev.value, ensure_ascii=False, default=str)
            if len(val_json) > 800:
                val_json = val_json[:800] + "... (truncated)"
            lines.append(f"  value: {val_json}")

            if ev.location:
                lines.append(
                    f"  location: "
                    f"{json.dumps(ev.location, ensure_ascii=False, default=str)}"
                )
            lines.append("")

        return "\n".join(lines)