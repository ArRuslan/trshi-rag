from pathlib import Path
from typing import Any, cast

import docutils.nodes
import sphinx.addnodes
from docutils.nodes import Node
from sphinx.application import Sphinx


class IdkVisitor(docutils.nodes.GenericNodeVisitor):
    def __init__(self, document: docutils.nodes.document) -> None:
        super().__init__(document)
        self.symbols: list[list[str]] = []
        self._in_desc = False
        self._in_content = False
        self._in_signature = False

    def visit_Text(self, node: docutils.nodes.Text) -> Any:
        if not self._in_content:
            return
        self.symbols[-1].append(node.replace("\n", ""))

    def default_visit(self, node: docutils.nodes.Node) -> None:
        node = cast(docutils.nodes.Element, node)

        cls = type(node)
        if cls is sphinx.addnodes.desc:
            self._in_desc = True

            self.symbols.append([
                f"Source: {self.document['source'].split('Doc/')[1]} ",
            ])

            signatures = [element for element in node.children if isinstance(element, sphinx.addnodes.desc_signature)]

            if signatures:
                self.symbols[-1].append(node["desctype"].title())
                self.symbols[-1].append(" ")
                self.symbols[-1].append(signatures[0]["module"])
                self.symbols[-1].append(".")
                self.symbols[-1].append(signatures[0]["fullname"])

                self.symbols[-1].append(" Signatures: " if len(signatures) > 1 else " Signature: ")

                for signature in signatures:
                    self.symbols[-1].append(signature.astext())
                    self.symbols[-1].append(" ")

        elif self._in_desc and cls is sphinx.addnodes.desc_signature:
            raise docutils.nodes.SkipNode
        elif self._in_desc and cls is sphinx.addnodes.desc_content:
            self._in_content = True

        # print(type(node))

    def default_departure(self, node: Node) -> None:
        cls = type(node)
        if cls is sphinx.addnodes.desc:
            self._in_desc = False
            self._in_content = False
        elif self._in_desc and cls is sphinx.addnodes.desc_content:
            self._in_content = False


def _on_doctree_resolved(app: Sphinx, doc: docutils.nodes.document, name: str) -> None:
    if name != "library/bisect":
        return

    print(doc.pformat())

    visitor = IdkVisitor(doc)
    doc.walk(visitor)

    for desc in visitor.symbols:
        print("".join(desc))


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
