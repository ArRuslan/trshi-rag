from pathlib import Path
from typing import Any, cast, TypedDict

import docutils.nodes
import sphinx.addnodes
from docutils.nodes import Node
from sphinx.application import Sphinx


class RagPart(TypedDict):
    document: str
    section: str
    signatures: list[str]
    text: str


class IdkVisitor(docutils.nodes.GenericNodeVisitor):
    def __init__(self, document: docutils.nodes.document) -> None:
        super().__init__(document)
        self.parts: list[tuple[RagPart, list[str]]] = []
        self._in_desc = False
        self._in_content = False
        self._in_section = False
        self._was_in_section = False
        self._in_paragraph = False

    def visit_Text(self, node: docutils.nodes.Text) -> Any:
        if not self._in_content and not self._in_paragraph:
            return
        self.parts[-1][1].append(node.replace("\n", ""))

    def visit_literal_block(self, node: docutils.nodes.literal_block) -> Any:
        if not self._in_content and not self._in_section:
            return
        assert len(node.children) == 1
        assert isinstance(node.children[0], docutils.nodes.Text)
        self.parts[-1][1].append("\n")
        self.parts[-1][1].append(cast(docutils.nodes.Text, node.children[0]))
        raise docutils.nodes.SkipNode

    def visit_section(self, node: docutils.nodes.section) -> Any:
        assert not self._in_desc
        self._in_section = True

        titles = list(node.findall(docutils.nodes.title))

        text_parts = []
        part_info: RagPart = {
            "document": self.document["source"].split("Doc/")[1],
            "section": titles[0].astext(),
            "signatures": [],
            "text": "",
        }

        self.parts.append((part_info, text_parts))

    def depart_section(self, node: docutils.nodes.section) -> Any:
        self._in_section = False
        self._join_text()

    def visit_paragraph(self, node: docutils.nodes.paragraph) -> Any:
        if not self._in_section:
            return
        self._in_paragraph = True

    def depart_paragraph(self, node: docutils.nodes.paragraph) -> Any:
        if not self._in_section:
            return
        self._in_paragraph = False

    def default_visit(self, node: docutils.nodes.Node) -> None:
        node = cast(docutils.nodes.Element, node)

        cls = type(node)
        if cls is sphinx.addnodes.desc:
            self._was_in_section = self._in_section
            self._in_section = False

            if self._was_in_section:
                self._join_text()

            self._in_desc = True

            text_parts = []
            part_info: RagPart = {
                "document": self.document["source"].split("Doc/")[1],
                "section": "",
                "signatures": [],
                "text": "",
            }

            self.parts.append((part_info, text_parts))

            signatures = [element for element in node.children if isinstance(element, sphinx.addnodes.desc_signature)]

            if signatures:
                part_info["section"] = signatures[0].astext()

                text_parts.append(node["desctype"].title())
                text_parts.append(" ")
                text_parts.append(signatures[0]["module"])
                text_parts.append(".")
                text_parts.append(signatures[0]["fullname"])

                for signature in signatures:
                    part_info["signatures"].append(signature.astext())

        elif self._in_desc and cls is sphinx.addnodes.desc_signature:
            raise docutils.nodes.SkipNode
        elif self._in_desc and cls is sphinx.addnodes.desc_content:
            self._in_content = True

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
        elif self._in_desc and cls is sphinx.addnodes.desc_content:
            self._in_content = False


def _on_doctree_resolved(app: Sphinx, doc: docutils.nodes.document, name: str) -> None:
    if name != "library/bisect":
        return

    print(doc.pformat())

    visitor = IdkVisitor(doc)
    doc.walkabout(visitor)

    for part in visitor.parts:
        print(part)


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
