from collections import defaultdict
from typing import Optional

from .models import EvidenceItem, FeverousSample
from .utils.annotation_processor import AnnotationProcessor
from .utils.wiki_page import WikiPage
from .database.feverous_db import FeverousDB
from ..base import Dataset
from ...utils import normalize_feverous_label, wiki_to_plain_text

class Feverous(Dataset):
    def __init__(self,
                 annotations: AnnotationProcessor,
                 wiki_db: Optional[FeverousDB] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.annotations = annotations
        self.wiki_db = wiki_db

    @classmethod
    def from_path(cls, dataset_path: str, db_path: Optional[str] = None):
        anno_processor = AnnotationProcessor(dataset_path)
        wiki_db = FeverousDB(db_path)

        return cls(anno_processor, wiki_db, claims=None)

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        for annotation in self.annotations:
            claim = annotation.get_claim()

            try:
                challenge = annotation.get_challenge()
                label = normalize_feverous_label(annotation.get_verdict())
                context_dicts = annotation.get_context(flat=True)
                evidences = annotation.get_evidence(flat=True)

                evidence_list = []
                for i, ev_id in enumerate(evidences):
                    wiki_doc = ev_id.split('_')[0]
                    evidence_id = '_'.join(ev_id.split('_')[1:])

                    page_json = self.wiki_db.get_doc_json(wiki_doc)
                    wiki_page = WikiPage(wiki_doc, page_json)

                    source = wiki_page.get_title_content() or wiki_doc

                    # sentence: handled implicitly via get_element_by_id (sentence in page_items)
                    # title: explicit handling needed (title not in page_items)
                    # cell/header_cell, item, table_caption: need specialized getters
                    content = wiki_page.get_element_by_id(evidence_id)
                    if "title" in evidence_id:
                        content = "Title: " + wiki_page.get_title_content()
                    elif "cell" in evidence_id:
                        content = "Cell: " + wiki_page.get_cell_content(evidence_id)
                    elif "item" in evidence_id:
                        content = "Item: " + wiki_page.get_item_by_id(evidence_id)
                    elif "table_caption" in evidence_id:
                        content = "Table caption: " + str(
                            wiki_page.get_caption_content(evidence_id) or ""
                        )
                    ev_content = f"- Evidence {i+1}: {str(content)}"

                    # Build context for this specific evidence
                    ev_context_ids = context_dicts.get(ev_id, [])
                    context_parts = []
                    for j, ctx_id in enumerate(ev_context_ids):
                        ctx_wiki_doc = ctx_id.split('_')[0]
                        ctx_element_id = '_'.join(ctx_id.split('_')[1:])

                        ctx_page_json = self.wiki_db.get_doc_json(ctx_wiki_doc)
                        ctx_wiki_page = WikiPage(ctx_wiki_doc, ctx_page_json)

                        ctx_content = ctx_wiki_page.get_element_by_id(ctx_element_id)
                        if "title" in ctx_element_id:
                            ctx_content = "Title: " + ctx_wiki_page.get_title_content()
                        elif "cell" in ctx_element_id:
                            ctx_content = "Cell: " + ctx_wiki_page.get_cell_content(ctx_element_id)
                        elif "item" in ctx_element_id:
                            ctx_content = "Item: " + ctx_wiki_page.get_item_by_id(ctx_element_id)
                        elif "table_caption" in ctx_element_id:
                            ctx_content = "Table caption: " + str(
                                ctx_wiki_page.get_caption_content(ctx_element_id) or ""
                            )
                        context_parts.append(f"- Context {j+1}: {str(ctx_content)}")

                    evidence_list.append(EvidenceItem(
                        source=source,
                        content=ev_content,
                        context="\n".join(context_parts),
                    ))

            except:
                challenge = None
                evidence_list = None
                label = None

            yield FeverousSample(
                claim=claim,
                evidence=evidence_list,
                label=label,
            )


class FeverousEvidenceFormat(Dataset):
    """Dataset that formats FEVEROUS evidence for LLM fact-checking.

    Output format:
    - Evidence: grouped by wiki page, with sections [Source: <title>], Passages,
      Table(s), List items. Wiki markup (e.g. [[link|text]]) is stripped to plain text.
    - Context: one line per context element, plain text.

    Evidence marking:
    - Actual evidence elements are wrapped in **bold** (passages, table cells,
      table captions when caption-only, list items). Only a subset of content
      may be evidence; bold helps the model focus.
    - Prompt hint: add something like "The **bold** parts are the actual evidence."
      when using this format in your prompt.
    """

    def __init__(
        self,
        annotations: AnnotationProcessor,
        wiki_db: Optional[FeverousDB] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.annotations = annotations
        self.wiki_db = wiki_db

    @classmethod
    def from_path(cls, dataset_path: str, db_path: Optional[str] = None):
        anno_processor = AnnotationProcessor(dataset_path)
        wiki_db = FeverousDB(db_path)
        return cls(anno_processor, wiki_db, claims=None)

    def __getitem__(self, item):
        raise NotImplementedError

    def _format_table_as_markdown(
        self,
        wiki_page: WikiPage,
        table,
        evidence_cell_ids: set[str] | None = None,
    ) -> str:
        """Format WikiTable as markdown table with plain text cells.

        Format:
            | A | B | C |
            | --- | --- | --- |
            | x | **y** | z |

        Rationale:
        - Markdown tables are standard and easy for LLMs to parse.
        - [H] prefix on header cells is stripped so output is clean.
        - wiki_to_plain_text removes [[link|text]] markup from cell content.
        - evidence_cell_ids: cells in this set are wrapped in **bold** so the model
          knows which cells are the actual evidence (vs context).
        """
        evidence_cell_ids = evidence_cell_ids or set()
        lines = []
        for row in table.rows:
            cells = []
            for cell in row.row:
                raw = str(cell)
                if raw.startswith("[H] "):
                    raw = raw[4:]
                content = wiki_to_plain_text(raw)
                if cell.name in evidence_cell_ids:
                    content = f"**{content}**"
                cells.append(content)
            lines.append("| " + " | ".join(cells) + " |")
        if lines:
            col_count = len(lines[0].split("|")[1:-1])
            header = "| " + " | ".join(["---"] * col_count) + " |"
            return "\n".join([lines[0], header] + lines[1:])
        return ""

    def _parse_table_id_from_cell_id(self, element_id: str) -> str:
        """Parse table_id from cell_id or header_cell_id.

        Format:
            cell_<table>_<row>_<col> -> table_<table>
            header_cell_<table>_<row>_<col> -> table_<table>

        Rationale:
        - Table index differs by position: cell has table at index 1, header_cell at 2.
        - Used for deduplication (one table per evidence set) and grouping.
        """
        parts = element_id.split("_")
        if element_id.startswith("cell_"):
            return "table_" + parts[1] if len(parts) > 1 else "table_?"
        if element_id.startswith("header_cell_"):
            return "table_" + parts[2] if len(parts) > 2 else "table_?"
        return "table_?"

    def _parse_table_id_from_caption_id(self, element_id: str) -> str:
        """Parse table_id from table_caption_id.

        Format:
            table_caption_<table> -> table_<table>

        Rationale:
        - Caption evidence references a table by its caption; we need table_id
          to fetch the full table and avoid duplicate displays when both cells
          and caption are evidence.
        """
        if element_id.startswith("table_caption_"):
            parts = element_id.split("_")
            return "table_" + parts[2] if len(parts) > 2 else "table_?"
        return "table_?"

    def _group_evidence_by_source(self, evidences: list[str]) -> dict[str, dict]:
        """Group evidence by wiki_doc and type.

        Format:
            {wiki_doc: {"sentences": [ev_id, ...], "cells": [(ev_id, table_id), ...],
             "items": [ev_id, ...], "captions": [(ev_id, table_id), ...]}}

        Rationale:
        - Group by page: load each page once and render all evidence from it.
        - Separate by type: sentences, tables (cells/captions), lists have different
          render logic and presentation order (passages first, then tables, then items).
        - Cells and captions store table_id for deduplication and table lookup.
        """
        by_source: dict[str, dict[str, list]] = defaultdict(
            lambda: {"sentences": [], "cells": [], "items": [], "captions": []}
        )
        for ev_id in evidences:
            parts = ev_id.split("_", 1)
            if len(parts) < 2:
                continue
            wiki_doc, element_id = parts[0], parts[1]
            if "sentence_" in element_id:
                by_source[wiki_doc]["sentences"].append(ev_id)
            elif "cell_" in element_id or "header_cell_" in element_id:
                table_id = self._parse_table_id_from_cell_id(element_id)
                by_source[wiki_doc]["cells"].append((ev_id, table_id))
            elif "table_caption_" in element_id:
                table_id = self._parse_table_id_from_caption_id(element_id)
                by_source[wiki_doc]["captions"].append((ev_id, table_id))
            elif "item_" in element_id:
                by_source[wiki_doc]["items"].append(ev_id)
        return dict(by_source)

    def _format_evidence_str(self, evidences: list[str]) -> str:
        """Build structured evidence string from grouped evidence.

        Format:
            [Source: <page title>]

            Passages:
            - <sentence 1>
            - <sentence 2>

            Table: <caption or empty>
            | col1 | col2 |
            | --- | --- |
            | ... |

            List items:
            - <item 1>
            - <item 2>

            (Sections separated by "\\n\\n". Multiple pages separated by "\\n\\n".)

        Rationale:
        - Source first: helps model attribute evidence to the right page.
        - Passages before tables before items: matches typical reading order.
        - One table per table_id: cells and captions from same table shown once.
        - Caption-only tables: when evidence is only table_caption, still show full
          table so the model has context.
        - Bold marking: evidence cells/sentences/items are **bold** so the model
          knows which parts support the claim (vs context).
        """
        grouped = self._group_evidence_by_source(evidences)
        sections = []
        for wiki_doc, ev_dict in grouped.items():
            page_json = self.wiki_db.get_doc_json(wiki_doc)
            wiki_page = WikiPage(wiki_doc, page_json)
            source = wiki_to_plain_text(wiki_page.get_title_content()) or wiki_doc
            parts = []

            if ev_dict["sentences"]:
                parts.append("Passages:")
                for ev_id in ev_dict["sentences"]:
                    element_id = ev_id.split("_", 1)[1]
                    elem = wiki_page.get_element_by_id(element_id)
                    if elem:
                        content = wiki_to_plain_text(str(elem))
                        parts.append(f"- **{content}**")
                parts.append("")

            table_evidence_cells: dict[str, set[str]] = defaultdict(set)
            for ev_id, table_id in ev_dict["cells"]:
                element_id = ev_id.split("_", 1)[1]
                table_evidence_cells[table_id].add(element_id)

            caption_evidence_tables = {t_id for _, t_id in ev_dict["captions"]}

            tables_done = set()
            if ev_dict["cells"]:
                for ev_id, table_id in ev_dict["cells"]:
                    if table_id in tables_done:
                        continue
                    tables_done.add(table_id)
                    element_id = ev_id.split("_", 1)[1]
                    table = wiki_page.get_table_from_cell_id(element_id)
                    if table:
                        caption = wiki_to_plain_text(table.get_table_caption())
                        if caption and table_id in caption_evidence_tables:
                            table_label = f"Table: **{caption}**"
                        else:
                            table_label = f"Table: {caption}" if caption else "Table"
                        parts.append(table_label)
                        evidence_cell_ids = table_evidence_cells[table_id]
                        parts.append(
                            self._format_table_as_markdown(
                                wiki_page, table, evidence_cell_ids
                            )
                        )
                        parts.append("")

            if ev_dict["captions"]:
                for ev_id, table_id in ev_dict["captions"]:
                    if table_id in tables_done:
                        continue
                    tables_done.add(table_id)
                    element_id = ev_id.split("_", 1)[1]
                    table = wiki_page.get_element_by_id(table_id)
                    if table:
                        caption = wiki_to_plain_text(
                            wiki_page.get_caption_content(element_id) or ""
                        )
                        table_label = (
                            f"Table: **{caption}**" if caption else "Table"
                        )
                        parts.append(table_label)
                        parts.append(
                            self._format_table_as_markdown(wiki_page, table)
                        )
                        parts.append("")

            if ev_dict["items"]:
                parts.append("List items:")
                for ev_id in ev_dict["items"]:
                    element_id = ev_id.split("_", 1)[1]
                    content = wiki_page.get_item_by_id(element_id)
                    if content is not None:
                        parts.append(
                            f"- **{wiki_to_plain_text(str(content))}**"
                        )
                parts.append("")

            sections.append((source, "\n".join(parts).strip()))
        return sections

    def _format_context_str(self, context_dicts: dict) -> str:
        """Format context with plain text.

        Format:
            One line per context element, newline-separated.
            Content is normalized via wiki_to_plain_text (no markup).

        Supports: title, cell, header_cell, item, table_caption, sentence.
        Rationale:
        - Context provides supporting elements (e.g. section headers, surrounding
          cells); flattening to lines keeps it compact.
        - Same element-type dispatch as evidence: title/table_caption need
          specialized getters; sentence uses get_element_by_id.
        """
        parts = []
        for ev_key, contexts in context_dicts.items():
            for ctx_id in contexts:
                wiki_doc = ctx_id.split("_")[0]
                element_id = "_".join(ctx_id.split("_")[1:])
                page_json = self.wiki_db.get_doc_json(wiki_doc)
                wiki_page = WikiPage(wiki_doc, page_json)
                if "title" in element_id:
                    content = wiki_page.get_title_content()
                elif "cell" in element_id:
                    content = wiki_page.get_cell_content(element_id)
                elif "item" in element_id:
                    content = wiki_page.get_item_by_id(element_id)
                elif "table_caption_" in element_id:
                    content = wiki_page.get_caption_content(element_id)
                elif "sentence_" in element_id:
                    elem = wiki_page.get_element_by_id(element_id)
                    content = str(elem) if elem else ""
                else:
                    content = ""
                if content:
                    parts.append(wiki_to_plain_text(str(content)))
        return "\n".join(parts) if parts else ""

    def __iter__(self):
        for annotation in self.annotations:
            claim = annotation.get_claim()
            try:
                label = normalize_feverous_label(annotation.get_verdict())
                context_dicts = annotation.get_context(flat=True)
                evidences = annotation.get_evidence(flat=True)

                evidence_list = []
                for ev_id in evidences:
                    ev_sections = self._format_evidence_str([ev_id])
                    ev_context_ids = context_dicts.get(ev_id, [])
                    ev_context = (
                        self._format_context_str({ev_id: ev_context_ids})
                        if ev_context_ids
                        else ""
                    )
                    # Each ev_sections entry is (source, content)
                    for source, content in ev_sections:
                        evidence_list.append(EvidenceItem(
                            source=source,
                            content=content,
                            context=ev_context,
                        ))
            except Exception:
                evidence_list = None
                label = None

            yield FeverousSample(
                claim=claim,
                evidence=evidence_list,
                label=label,
            )


class FeverousStructuredFormat(Dataset):
    """Dataset that formats FEVEROUS evidence as structured XML-tagged blocks.

    Improvements over FeverousEvidenceFormat:
    1. **Grouped by source**: All evidence from the same Wikipedia page is
       merged into a single <source> block, avoiding duplication.
    2. **XML tags for structure**: Uses <source>, <passage>, <table>, <list>,
       <context> tags so the LLM can parse evidence types unambiguously.
    3. **Inline context**: Context appears inside the same <source> block as
       its evidence, making the relationship explicit.
    4. **Evidence markers via [EVIDENCE] prefix**: Clearer than bold for
       marking which parts are the actual evidence vs surrounding context.
    5. **Deduplicated tables**: Multiple cells from the same table produce one
       table render with marked cells, not N separate renders.

    Output format example:
        <source title="Albert Einstein">
        <passages>
        [EVIDENCE] Albert Einstein was born on March 14, 1879.
        [EVIDENCE] He developed the theory of general relativity.
        </passages>

        <table caption="Nobel Prize winners">
        | Year | Laureate | Field |
        | --- | --- | --- |
        | 1921 | [EVIDENCE] Albert Einstein | [EVIDENCE] Physics |
        </table>

        <context>
        Einstein was a theoretical physicist.
        He is best known for his mass-energy equivalence formula.
        </context>
        </source>
    """

    def __init__(
        self,
        annotations: AnnotationProcessor,
        wiki_db: Optional[FeverousDB] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.annotations = annotations
        self.wiki_db = wiki_db

    @classmethod
    def from_path(cls, dataset_path: str, db_path: Optional[str] = None):
        anno_processor = AnnotationProcessor(dataset_path)
        wiki_db = FeverousDB(db_path)
        return cls(anno_processor, wiki_db, claims=None)

    def __getitem__(self, item):
        raise NotImplementedError

    # ── table formatting ──────────────────────────────────────────────

    def _format_table_as_markdown(
        self,
        wiki_page: WikiPage,
        table,
        evidence_cell_ids: set[str] | None = None,
    ) -> str:
        """Render table as markdown. Evidence cells get [EVIDENCE] prefix."""
        evidence_cell_ids = evidence_cell_ids or set()
        lines = []
        for row in table.rows:
            cells = []
            for cell in row.row:
                raw = str(cell)
                if raw.startswith("[H] "):
                    raw = raw[4:]
                content = wiki_to_plain_text(raw)
                if cell.name in evidence_cell_ids:
                    content = f"[EVIDENCE] {content}"
                cells.append(content)
            lines.append("| " + " | ".join(cells) + " |")
        if lines:
            col_count = len(lines[0].split("|")[1:-1])
            header = "| " + " | ".join(["---"] * col_count) + " |"
            return "\n".join([lines[0], header] + lines[1:])
        return ""

    # ── helpers ────────────────────────────────────────────────────────

    def _parse_table_id_from_cell_id(self, element_id: str) -> str:
        parts = element_id.split("_")
        if element_id.startswith("cell_"):
            return "table_" + parts[1] if len(parts) > 1 else "table_?"
        if element_id.startswith("header_cell_"):
            return "table_" + parts[2] if len(parts) > 2 else "table_?"
        return "table_?"

    def _parse_table_id_from_caption_id(self, element_id: str) -> str:
        if element_id.startswith("table_caption_"):
            parts = element_id.split("_")
            return "table_" + parts[2] if len(parts) > 2 else "table_?"
        return "table_?"

    # ── grouping ──────────────────────────────────────────────────────

    def _group_evidence_by_source(
        self, evidences: list[str]
    ) -> dict[str, dict[str, list]]:
        by_source: dict[str, dict[str, list]] = defaultdict(
            lambda: {"sentences": [], "cells": [], "items": [], "captions": []}
        )
        for ev_id in evidences:
            parts = ev_id.split("_", 1)
            if len(parts) < 2:
                continue
            wiki_doc, element_id = parts[0], parts[1]
            if "sentence_" in element_id:
                by_source[wiki_doc]["sentences"].append(ev_id)
            elif "cell_" in element_id or "header_cell_" in element_id:
                table_id = self._parse_table_id_from_cell_id(element_id)
                by_source[wiki_doc]["cells"].append((ev_id, table_id))
            elif "table_caption_" in element_id:
                table_id = self._parse_table_id_from_caption_id(element_id)
                by_source[wiki_doc]["captions"].append((ev_id, table_id))
            elif "item_" in element_id:
                by_source[wiki_doc]["items"].append(ev_id)
        return dict(by_source)

    def _group_context_by_source(
        self, context_dicts: dict, evidence_ids: list[str]
    ) -> dict[str, list[str]]:
        """Group context text by wiki_doc, deduplicating against evidence."""
        evidence_set = set(evidence_ids)
        by_source: dict[str, list[str]] = defaultdict(list)
        for _ev_key, contexts in context_dicts.items():
            for ctx_id in contexts:
                if ctx_id in evidence_set:
                    continue  # already shown as evidence
                wiki_doc = ctx_id.split("_")[0]
                element_id = "_".join(ctx_id.split("_")[1:])
                page_json = self.wiki_db.get_doc_json(wiki_doc)
                wiki_page = WikiPage(wiki_doc, page_json)
                content = self._get_element_content(wiki_page, element_id)
                if content:
                    text = wiki_to_plain_text(str(content))
                    if text not in by_source[wiki_doc]:
                        by_source[wiki_doc].append(text)
        return dict(by_source)

    def _get_element_content(self, wiki_page: WikiPage, element_id: str):
        """Resolve element content by type."""
        if "title" in element_id:
            return wiki_page.get_title_content()
        elif "cell" in element_id:
            return wiki_page.get_cell_content(element_id)
        elif "item" in element_id:
            return wiki_page.get_item_by_id(element_id)
        elif "table_caption_" in element_id:
            return wiki_page.get_caption_content(element_id)
        elif "sentence_" in element_id:
            elem = wiki_page.get_element_by_id(element_id)
            return str(elem) if elem else ""
        return ""

    # ── main rendering ────────────────────────────────────────────────

    def _render_source_block(
        self,
        wiki_doc: str,
        ev_dict: dict[str, list],
        context_lines: list[str] | None,
    ) -> str:
        """Render a single <source> block with all evidence types + context."""
        page_json = self.wiki_db.get_doc_json(wiki_doc)
        wiki_page = WikiPage(wiki_doc, page_json)
        source_title = wiki_to_plain_text(
            wiki_page.get_title_content()
        ) or wiki_doc
        parts = []

        # ── passages ──
        if ev_dict["sentences"]:
            passage_lines = []
            for ev_id in ev_dict["sentences"]:
                element_id = ev_id.split("_", 1)[1]
                elem = wiki_page.get_element_by_id(element_id)
                if elem:
                    content = wiki_to_plain_text(str(elem))
                    passage_lines.append(f"[EVIDENCE] {content}")
            if passage_lines:
                inner = "\n".join(passage_lines)
                parts.append(f"<passages>\n{inner}\n</passages>")

        # ── tables (deduplicated by table_id) ──
        table_evidence_cells: dict[str, set[str]] = defaultdict(set)
        for ev_id, table_id in ev_dict["cells"]:
            element_id = ev_id.split("_", 1)[1]
            table_evidence_cells[table_id].add(element_id)

        caption_evidence_tables = {t_id for _, t_id in ev_dict["captions"]}
        tables_done: set[str] = set()

        for ev_id, table_id in ev_dict["cells"]:
            if table_id in tables_done:
                continue
            tables_done.add(table_id)
            element_id = ev_id.split("_", 1)[1]
            table = wiki_page.get_table_from_cell_id(element_id)
            if table:
                caption = wiki_to_plain_text(table.get_table_caption())
                caption_attr = f' caption="{caption}"' if caption else ""
                if caption and table_id in caption_evidence_tables:
                    caption_attr = f' caption="[EVIDENCE] {caption}"'
                md = self._format_table_as_markdown(
                    wiki_page, table, table_evidence_cells[table_id]
                )
                parts.append(f"<table{caption_attr}>\n{md}\n</table>")

        for ev_id, table_id in ev_dict["captions"]:
            if table_id in tables_done:
                continue
            tables_done.add(table_id)
            element_id = ev_id.split("_", 1)[1]
            table = wiki_page.get_element_by_id(table_id)
            if table:
                caption = wiki_to_plain_text(
                    wiki_page.get_caption_content(element_id) or ""
                )
                caption_attr = (
                    f' caption="[EVIDENCE] {caption}"' if caption else ""
                )
                md = self._format_table_as_markdown(wiki_page, table)
                parts.append(f"<table{caption_attr}>\n{md}\n</table>")

        # ── list items ──
        if ev_dict["items"]:
            item_lines = []
            for ev_id in ev_dict["items"]:
                element_id = ev_id.split("_", 1)[1]
                content = wiki_page.get_item_by_id(element_id)
                if content is not None:
                    item_lines.append(
                        f"[EVIDENCE] {wiki_to_plain_text(str(content))}"
                    )
            if item_lines:
                inner = "\n".join(item_lines)
                parts.append(f"<list>\n{inner}\n</list>")

        # ── context ──
        if context_lines:
            inner = "\n".join(context_lines)
            parts.append(f"<context>\n{inner}\n</context>")

        body = "\n\n".join(parts)
        return f'<source title="{source_title}">\n{body}\n</source>'

    def _build_evidence_xml(
        self,
        evidences: list[str],
        context_dicts: dict,
    ) -> str:
        """Build complete structured evidence string from all evidence IDs."""
        grouped = self._group_evidence_by_source(evidences)
        ctx_by_source = self._group_context_by_source(context_dicts, evidences)

        blocks = []
        for wiki_doc, ev_dict in grouped.items():
            context_lines = ctx_by_source.get(wiki_doc)
            block = self._render_source_block(wiki_doc, ev_dict, context_lines)
            blocks.append(block)
        return "\n\n".join(blocks)

    # ── iteration ─────────────────────────────────────────────────────

    def __iter__(self):
        for annotation in self.annotations:
            claim = annotation.get_claim()
            try:
                label = normalize_feverous_label(annotation.get_verdict())
                context_dicts = annotation.get_context(flat=True)
                evidences = annotation.get_evidence(flat=True)

                evidence_xml = self._build_evidence_xml(evidences, context_dicts)
            except Exception:
                evidence_xml = None
                label = None

            yield FeverousSample(
                claim=claim,
                evidence=evidence_xml,
                label=label,
            )