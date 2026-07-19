# Isolated PyroFinder Sites applications

This directory stores standalone web applications next to the Streamlit project.
Nothing under `sites/` is imported by `app.py`, `pages/`, or `src/`, and Node
dependencies and build output must remain inside the selected site directory.

## Current presentation version

- Folder: `pyrofinder-hanging-tree-live/`
- Production: https://pyrofinder-hanging-tree-v5.lisaborisclark.chatgpt.site
- Canonical source: `AzarovBoris/PyroFinderApp` tag
  `sites-canonical-2026-07-19` at commit `854b3f37a1f93425e419033023dd07f5ae69cfa8`
- Import method: Git subtree, so the snapshot can be compared and updated without
  coupling it to the Streamlit runtime

Run Node commands only from `sites/pyrofinder-hanging-tree-live/`.

## Archive

`hanging-tree-multicamera-v5/` is the former embedded snapshot. It remains in Git
for historical comparison and rollback only; do not present or deploy it as the
current app.

## Boundary rule

Updating a Sites snapshot must not change Streamlit application code, Python
dependencies, model weights, Streamlit secrets, or the public Streamlit
deployment. The nested `.openai/hosting.json` belongs only to the standalone
Sites project.
