# Documentation

The site has two halves:

- **API reference** — generated from the source by [pdoc](https://pdoc.dev).
  Nothing to maintain by hand; it follows the code.
- **Narrative pages** — `index.md` and `user_guide.md` in this folder, written
  by hand and converted at build time.

## Building locally

```bash
python docs/build_docs.py            # -> site/
python -m http.server -d site 8000   # then open http://localhost:8000
```

`site/` is gitignored; CI rebuilds it on every push to `main`.

## Editing

| To change | Edit |
| --- | --- |
| Landing page | `index.md` |
| User guide prose | `user_guide.md` |
| Page styling / nav | `_page_template.html` |
| Which modules are documented | `MODULES` in `build_docs.py` |

Anything in the API reference comes from docstrings in `src/` — edit the
docstring, not the generated HTML. Docstrings are parsed as Google style, so
`Args:` / `Returns:` sections render as structured blocks.

Sections marked with `<div class="todo">` in `user_guide.md` are placeholders
flagging where project-specific guidance would help.

## The `--mock` flag

`build_docs.py --mock` stubs the heavy runtime libraries (torch, shap, captum,
stable-baselines3, ...) instead of importing them, which is what lets the CI job
install seven packages rather than forty. Verified to produce output identical
to a build against the full `requirements.txt`.

Use it when you want to build the docs without the full scientific stack
installed. Two details make it work:

- Mocked attributes are real classes, because libraries introspect them —
  scipy calls `issubclass(cls, torch.Tensor)` whenever `torch` is in
  `sys.modules`, which raises `TypeError` against a non-class.
- A meta-path finder serves whole subtrees, since `import torch.nn` asks the
  import system for that submodule specifically.

If a *new* heavy dependency starts breaking the mocked build, add it to
`MOCK_PACKAGES`; if a library needs to be real (typically because its types
appear in annotations), add it to `requirements-docs.txt` instead.

## Deployment

`.github/workflows/docs.yml` builds and publishes to GitHub Pages on every push
to `main`. It needs **Settings → Pages → Source: GitHub Actions** set once in
the repository.

The build pins Python 3.10: on 3.13 the standard library exposes
`pathlib._local.Path` in signatures instead of `pathlib.Path`.

## Note on the lazy facade

`src/__init__.py` resolves its public names lazily via PEP 562 `__getattr__`, so
they never appear in `__dict__` until accessed and pdoc reports all 32 as
unresolvable. `build_docs.py` materialises them in memory before rendering; the
source stays lazy.
