---
title: CPAC Road Obstacle Detection
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: agpl-3.0
---

Gradio front end for the CPAC detector. To deploy, create a Gradio Space and copy `app.py` and
`requirements.txt` from this folder. The app downloads the weights from the GitHub release
(`v1.0.0`) on first start. The weights are Ultralytics models, so the Space is AGPL-3.0.

Run locally: `pip install -r requirements.txt && python app.py`.
