from pathlib import Path

import docutils.nodes
from sphinx.application import Sphinx


def _on_doctree_resolved(app: Sphinx, doc: docutils.nodes.document, name: str) -> None:
    if name != "library/bisect":
        return
    # for node in doc.findall(docutils.nodes.paragraph):
    #     print(node)


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
