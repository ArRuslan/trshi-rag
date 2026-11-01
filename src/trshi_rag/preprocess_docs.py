from pathlib import Path
from typing import Any, cast, TypedDict

import docutils.nodes
import sphinx.addnodes
from docutils.nodes import Node
from sphinx.application import Sphinx


class RagPart(TypedDict):
    document: str
    section: str
    position: int
    span_start: int
    span_end: int
    text: str


class IdkVisitor(docutils.nodes.GenericNodeVisitor):
    def __init__(self, document: docutils.nodes.document) -> None:
        super().__init__(document)
        self.parts: list[tuple[RagPart, list[str]]] = []
        self._in_desc = False
        self._in_content = False
        self._in_signature = False

    def visit_Text(self, node: docutils.nodes.Text) -> Any:
        if not self._in_content:
            return
        self.parts[-1][1].append(node.replace("\n", ""))

    def default_visit(self, node: docutils.nodes.Node) -> None:
        node = cast(docutils.nodes.Element, node)

        cls = type(node)
        if cls is sphinx.addnodes.desc:
            self._in_desc = True

            text_parts = []
            part_info = {
                "document": self.document["source"].split("Doc/")[1],
                "section": "",
                "position": len(self.parts),
                "span_start": 0,
                "span_end": 0,
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

                text_parts.append(" Signatures: " if len(signatures) > 1 else " Signature: ")

                for signature in signatures:
                    text_parts.append(signature.astext())
                    text_parts.append(" ")

        elif self._in_desc and cls is sphinx.addnodes.desc_signature:
            raise docutils.nodes.SkipNode
        elif self._in_desc and cls is sphinx.addnodes.desc_content:
            self._in_content = True

    def default_departure(self, node: Node) -> None:
        cls = type(node)

        if cls is sphinx.addnodes.desc:
            self._in_desc = False
            self._in_content = False
            part_info, text_parts = self.parts[-1]
            part_info["text"] = "".join(text_parts)
            text_parts.clear()
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
