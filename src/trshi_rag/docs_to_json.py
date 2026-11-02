import json
from pathlib import Path
from typing import Any, cast, TypedDict, Iterable, TypeVar

import docutils.nodes
import sphinx.addnodes
from docutils.nodes import Node
from sphinx.application import Sphinx


class RagPartSymbol(TypedDict):
    type: str
    signatures: list[str]
    module: str
    class_: str
    function: str
    attribute: str
    other: str


class RagPart(TypedDict):
    document: str
    section: str
    symbol: RagPartSymbol | None
    text: str


T = TypeVar("T", bound=Node)


def find_first_node(node: Node, child_cls: type[T]) -> Iterable[T]:
    for child in node.children:
        if isinstance(child, child_cls):
            yield child
        else:
            yield from find_first_node(child, child_cls)


class IdkVisitor(docutils.nodes.GenericNodeVisitor):
    def __init__(self, document: docutils.nodes.document) -> None:
        super().__init__(document)
        self.parts: list[tuple[RagPart, list[str]]] = []
        self._in_desc = False
        self._in_content = False
        self._in_section = False
        self._in_glossary = False
        self._was_in_section = False
        self._in_paragraph = False
        self._class_stack: list[tuple[sphinx.addnodes.desc, str]] = []

    def visit_Text(self, node: docutils.nodes.Text) -> Any:
        if not self._in_content and not self._in_paragraph:
            return
        self.parts[-1][1].append(node.replace("\n", ""))

    def visit_literal_block(self, node: docutils.nodes.literal_block) -> Any:
        if not self._in_content and not self._in_section:
            return
        self.parts[-1][1].append("\n")
        self.parts[-1][1].append(node.astext())
        raise docutils.nodes.SkipNode

    def visit_section(self, node: docutils.nodes.section) -> Any:
        assert not self._in_desc

        self._was_in_section = self._in_section
        self._in_section = True
        if self._was_in_section:
            self._join_text()

        titles = list(node.findall(docutils.nodes.title))

        text_parts = []
        part_info: RagPart = {
            "document": self.document["source"].split("Doc/")[1],
            "section": titles[0].astext(),
            "symbol": None,
            "text": "",
        }

        self.parts.append((part_info, text_parts))

    def depart_section(self, node: docutils.nodes.section) -> Any:
        self._in_section = False
        self._join_text()
        if self._was_in_section:
            self._in_section = True
            self._was_in_section = False
            if isinstance(node.parent, docutils.nodes.section):
                self.visit_section(node.parent)

    def visit_paragraph(self, node: docutils.nodes.paragraph) -> Any:
        if not self._in_section:
            return
        self._in_paragraph = True

    def depart_paragraph(self, node: docutils.nodes.paragraph) -> Any:
        if not self._in_section:
            return
        self._in_paragraph = False

    def visit_definition_list_item(self, node: docutils.nodes.definition_list_item) -> Any:
        if not self._in_glossary:
            return

        text_parts = []
        part_info: RagPart = {
            "document": self.document["source"].split("Doc/")[1],
            "section": "",
            "symbol": None,
            "text": "",
        }

        self.parts.append((part_info, text_parts))

        terms = [element for element in node.children if isinstance(element, docutils.nodes.term)]
        part_info["section"] = f"Glossary - \"{terms[0].astext()}\""

        text_parts.append(f"Definition of {', '.join(term.astext() for term in terms)}: ")

    def depart_definition_list_item(self, node: docutils.nodes.definition_list_item) -> Any:
        if not self._in_glossary:
            return
        self._join_text()

    def visit_table(self, node: docutils.nodes.table) -> Any:
        if not self._in_content and not self._in_section:
            raise docutils.nodes.SkipNode

        heads = list(node.findall(docutils.nodes.thead))
        bodies = list(node.findall(docutils.nodes.tbody))

        if len(heads) != 1 or len(bodies) != 1:
            raise docutils.nodes.SkipNode

        head = heads[0]
        body = bodies[0]

        cols = [entry.astext() for entry in next(head.findall(docutils.nodes.row)).findall(docutils.nodes.entry)]
        if len(cols) <= 1:
            raise docutils.nodes.SkipNode

        _, text_parts = self.parts[-1]

        for row in find_first_node(body, docutils.nodes.row):
            wrote = False
            for col_name, col in zip(cols, find_first_node(row, docutils.nodes.entry)):
                value = col.astext().strip()
                if not value:
                    continue
                if wrote:
                    text_parts.append("; ")
                wrote = True
                text_parts.append(f"{col_name}: {value}")
            if wrote:
                text_parts.append(f"\n")

        raise docutils.nodes.SkipNode

    def default_visit(self, node: docutils.nodes.Node) -> None:
        node = cast(docutils.nodes.Element, node)

        cls = type(node)
        if cls is sphinx.addnodes.desc:
            assert not self._in_glossary

            self._was_in_section = self._in_section
            self._in_section = False

            if self._was_in_section or self._in_desc:
                self._join_text()

            self._in_desc = True

            desctype = node["desctype"]

            text_parts = []
            symbol_info: RagPartSymbol = {
                "type": desctype,
                "signatures": [],
                "module": "",
                "class_": "",
                "function": "",
                "attribute": "",
                "other": "",
            }
            part_info: RagPart = {
                "document": self.document["source"].split("Doc/")[1],
                "section": "",
                "symbol": symbol_info,
                "text": "",
            }

            self.parts.append((part_info, text_parts))

            signatures = [element for element in node.children if isinstance(element, sphinx.addnodes.desc_signature)]

            if signatures:
                part_info["section"] = signatures[0].astext()

                if name := signatures[0].get("fullname"):
                    if mod := signatures[0].get("module"):
                        name = f"{mod}.{name}"
                        symbol_info["module"] = mod

                    if desctype == "class":
                        self._class_stack.append((cast(sphinx.addnodes.desc, node), name))
                    else:
                        if self._class_stack and name.startswith(self._class_stack[-1][1]):
                            name = name[len(self._class_stack[-1][1]) + 1:]
                        symbol_info["attribute" if desctype == "attribute" else "function"] = name

                    symbol_info["class_"] = self._class_stack[-1][1] if self._class_stack else ""
                else:
                    ids = signatures[0]["ids"]
                    if ids:
                        symbol_info["other"] = ids[0]

                for signature in signatures:
                    symbol_info["signatures"].append(signature.astext())

        elif self._in_desc and cls is sphinx.addnodes.desc_signature:
            raise docutils.nodes.SkipNode
        elif self._in_desc and cls is sphinx.addnodes.desc_content:
            self._in_content = True
        elif cls is sphinx.addnodes.seealso:
            raise docutils.nodes.SkipNode
        elif cls is sphinx.addnodes.glossary:
            assert not self._in_desc
            self._in_glossary = True

    def _join_text(self) -> None:
        part_info, text_parts = self.parts[-1]
        if part_info["text"]:
            return
        part_info["text"] = " ".join(text_parts).replace("\x00", "")
        text_parts.clear()

    def default_departure(self, node: Node) -> None:
        cls = type(node)

        if cls is sphinx.addnodes.desc:
            self._in_desc = False
            self._in_content = False
            self._join_text()
            if self._was_in_section:
                self._was_in_section = False
                if self._in_section and isinstance(node.parent, docutils.nodes.section):
                    self.visit_section(node.parent)
            if self._class_stack and node == self._class_stack[-1][0]:
                self._class_stack.pop()
        elif self._in_desc and cls is sphinx.addnodes.desc_content:
            self._in_content = False
        elif cls is sphinx.addnodes.glossary:
            self._in_glossary = False

    def unknown_visit(self, node: Node) -> Any:
        if node.__class__.__name__ == "snippet_string_node":
            return self.visit_inline(cast(docutils.nodes.inline, node))

        print(f"unknown node: {node}")

    def unknown_departure(self, node: Node) -> Any:
        ...


def _on_doctree_resolved(app: Sphinx, doc: docutils.nodes.document, name: str) -> None:
    out_dir = Path(f"data/parts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # print(f"Processing {name}")

    visitor = IdkVisitor(doc)
    try:
        doc.walkabout(visitor)
    except Exception as e:
        print(f"error during {name} processing: {e}")
        return

    with open(out_dir / (name.replace("/", "_") + ".json"), "w") as f:
        parts = [part for part, _ in visitor.parts if part["text"]]
        json.dump(parts, f, indent=2)


def main() -> None:
    docs_root = Path("data/cpython/Doc")
    build_dir = "data/docs_build"
    build_path = Path(build_dir)

    app = Sphinx(
        srcdir=str(docs_root),
        confdir=str(docs_root),
        outdir=build_dir,
        doctreedir=str(build_path / "doctrees"),
        buildername="dummy",
        freshenv=False,
        parallel=8,
    )

    app.connect("doctree-resolved", _on_doctree_resolved)
    app.build()


if __name__ == "__main__":
    main()
