"""Tree-Edit-Distance-based Similarity (TEDS) for table recognition.

Adapted from the IBM PubTabNet reference implementation (Apache-2.0):
https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py

Simplifications vs the original:
- table structure is normalized to table > tr > td (thead/tbody flattened, th -> td)
- cell-content rename cost uses rapidfuzz normalized Levenshtein
- no multiprocessing

TEDS(pred, gt) = 1 - TED(tree_pred, tree_gt) / max(|tree_pred|, |tree_gt|)
where TED is the APTED tree edit distance. 1.0 = identical, 0.0 = nothing shared.
`structure_only=True` gives TEDS-S (ignores cell text, keeps spans/topology).
"""

from __future__ import annotations

import re
import unicodedata

from apted import APTED, Config
from lxml import html as lxml_html
from rapidfuzz.distance import Levenshtein


class TableNode:
    __slots__ = ("tag", "colspan", "rowspan", "content", "children")

    def __init__(self, tag, colspan=1, rowspan=1, content=""):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = []


class TedsConfig(Config):
    def __init__(self, structure_only=False):
        self.structure_only = structure_only

    def rename(self, node1, node2):
        if (
            node1.tag != node2.tag
            or node1.colspan != node2.colspan
            or node1.rowspan != node2.rowspan
        ):
            return 1.0
        if node1.tag == "td" and not self.structure_only:
            return Levenshtein.normalized_distance(node1.content, node2.content)
        return 0.0

    def children(self, node):
        return node.children


def _norm_cell(text):
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _int_attr(el, name):
    try:
        return max(1, int(el.get(name, "1")))
    except (TypeError, ValueError):
        return 1


def build_table_tree(html_str):
    """Parse an HTML string containing a <table> into a TableNode tree.

    Returns None if no table can be parsed.
    """
    try:
        root = lxml_html.fromstring(html_str)
    except Exception:
        return None
    table = root if root.tag == "table" else root.find(".//table")
    if table is None:
        return None
    tree = TableNode("table")
    for tr in table.iter("tr"):
        row = TableNode("tr")
        for cell in tr.iter("td", "th"):
            row.children.append(
                TableNode(
                    "td",
                    colspan=_int_attr(cell, "colspan"),
                    rowspan=_int_attr(cell, "rowspan"),
                    content=_norm_cell(cell.text_content()),
                )
            )
        tree.children.append(row)
    if not tree.children:
        return None
    return tree


def _tree_size(node):
    return 1 + sum(_tree_size(c) for c in node.children)


def teds(pred_html, gt_html, structure_only=False):
    """TEDS score in [0, 1]; None if the ground-truth table is unparseable."""
    gt_tree = build_table_tree(gt_html)
    if gt_tree is None:
        return None
    pred_tree = build_table_tree(pred_html) if pred_html else None
    if pred_tree is None:
        return 0.0
    distance = APTED(
        pred_tree, gt_tree, TedsConfig(structure_only=structure_only)
    ).compute_edit_distance()
    denom = max(_tree_size(pred_tree), _tree_size(gt_tree))
    return max(0.0, 1.0 - distance / denom)
