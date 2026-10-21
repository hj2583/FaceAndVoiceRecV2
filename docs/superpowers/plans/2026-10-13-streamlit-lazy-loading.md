# Streamlit Lazy Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Defer page imports/rendering until the corresponding tab is selected and defer Torch/ONNX runtime probing until the user requests it, without changing the seven-tab UI.

**Architecture:** Keep the existing hidden single-page `st.navigation` wrapper and the seven visible tabs. Feature-detect stateful `st.tabs` support; on supported versions rerun on selection and import/render only the open tab, while older versions use the current eager behavior. Keep runtime status unloaded until a “Check runtime” action is clicked.

**Tech Stack:** Python 3.10+, Streamlit `>=1.40`, `importlib`, `inspect`, session state, pytest.

## Global Constraints

- Preserve the current seven tab labels, order, and workflows.
- Preserve recognition, transcription, attendance, CPU/GPU, and model behavior.
- Keep runtime dependencies in `requirements.txt` and `requirements-gpu.txt`; do not duplicate them in `pyproject.toml`.
- Keep Python `>=3.10` and Streamlit `>=1.40`; do not raise version floors for lazy loading.
- Keep `sitecustomize.py` at the repository root.
- Do not add speculative services, repositories, or orchestration layers.
- Treat `data/` as live user data; do not stage, move, delete, or run runtime initialization against it.
- Preserve unrelated untracked root modules and `D live/.gitkeep`.
- Run validation against `tests/unit` and `tests/integration`; bare `pytest -q` currently collects tracked legacy root test copies alongside moved tests and fails with duplicate-module import errors. Do not delete or rewrite those copies in this feature.
- Do not change recognition thresholds, model configuration, database schema, or tab labels/Unicode.
- The app already caches DB initialization once per process after the separate SQLite plan; retain that call order.

---

## File Map

| File | Responsibility |
|---|---|
| `app/main.py` | Feature-detect stateful tabs, import the selected renderer on demand, expose the runtime-check control, preserve app navigation/startup. |
| `app/pages/*.py` | Existing per-tab renderers, imported only when selected where supported. |
| `app/ui/status.py` | Existing Torch/ONNX runtime probe; imported/called only after the runtime-check action. |
| `tests/unit/test_app_lazy_loading.py` | Stateful-tab lazy render, old-API fallback, and runtime probe state tests. |
| `tests/unit/test_app_navigation.py` | Existing labels/order/startup/cache assertions, updated only if interfaces change. |

## Task 1: Stateful Tabs and Lazy Page Imports

**Files:**
- Modify: `app/main.py`
- Create: `tests/unit/test_app_lazy_loading.py`

**Interfaces:**
- `TAB_PAGES: tuple[tuple[str, str], ...]` contains, in order, each existing visible label and the `app.pages.*` module path.
- `main()` calls `st.set_option("client.showSidebarNavigation", False)` after `st.set_page_config()` so Streamlit does not add automatic sidebar links for `app/pages/` renderer modules.
- `supports_stateful_tabs(tabs_callable: Callable[..., object]) -> bool` returns true only if `inspect.signature(tabs_callable)` exposes both `on_change` and `key`; `TypeError`/`ValueError` during signature inspection returns false.
- `_render_tab_pages(tab_containers: Sequence[object], module_paths: Sequence[str], *, active_only: bool, import_module: Callable[[str], object] = importlib.import_module) -> None` imports and renders each page under its tab context; when `active_only` is true it skips containers whose `.open` is false.
- `render_application() -> None` keeps the current title and seven tabs. On the stateful path it creates tabs with `on_change="rerun"` and a stable key; inside each tab context it imports/calls the page module only when `tab.open` is true. On the fallback path it uses `st.tabs(labels)` and imports/renders all page modules as before.
- Importing page modules remains deferred through `importlib.import_module`; `app/main.py` must not import any `app.pages.*` module at module scope.

- [ ] **Step 1: Add a fake-tab test for `_render_tab_pages` proving only the selected page imports and renders.** Create seven fake tab contexts with one `.open=True`, pass the exact seven module-path strings and `active_only=True`, inject a spy importer, and assert exactly the selected module's `render()` is called.

```python
from types import SimpleNamespace


class FakeTab:
    def __init__(self, opened: bool) -> None:
        self.open = opened

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_stateful_tabs_import_only_the_open_page():
    selected = "app.pages.video"
    module_paths = (
        "app.pages.realtime",
        "app.pages.video",
        "app.pages.face_database",
        "app.pages.audio_logs",
        "app.pages.attendance",
        "app.pages.transcripts",
        "app.pages.voice_enrollment",
    )
    tabs = [FakeTab(module_path == selected) for module_path in module_paths]
    rendered = []

    def import_module(module_path):
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    _render_tab_pages(tabs, module_paths, active_only=True, import_module=import_module)

    assert rendered == [selected]
```

- [ ] **Step 2: Add a fallback test with seven fake tabs and `active_only=False`; assert all seven page renderers are called once. Add tests that `supports_stateful_tabs(tabs_callable)` returns true for a function accepting `on_change`/`key` and false for an old-signature function accepting only labels.**
- [ ] **Step 3: Run the new lazy-import tests and verify they fail because `_render_tab_pages` and the feature detector do not exist.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit/test_app_lazy_loading.py -q`

Expected: RED with missing helper/feature-detection symbols before implementation.

- [ ] **Step 4: Move the page import list to string module paths and implement `supports_stateful_tabs()` plus `_render_tab_pages()`.** Wire stateful `st.tabs` and eager fallback to those helpers. Keep tab labels, title, navigation, initialization, and render order unchanged.
- [ ] **Step 5: Run the lazy-import and navigation tests.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit/test_app_lazy_loading.py tests/unit/test_app_navigation.py -q`

Expected: both stateful and fallback paths pass, and app startup/navigation tests remain green.

- [ ] **Step 6: Commit stateful-tab lazy rendering.**

```powershell
git add app/main.py tests/unit/test_app_lazy_loading.py
git commit -m "feat: lazy load selected Streamlit tab"
```

## Task 2: On-Demand Runtime Status

**Files:**
- Modify: `app/main.py`
- Modify: `tests/unit/test_app_lazy_loading.py`
- Modify: `tests/unit/test_app_navigation.py` to model Streamlit button/session state and assert no probe or status captions before click.

**Interfaces:**
- The visible control is labeled `Check runtime`.
- Before activation, no Torch/ONNX probe is called and no runtime-status caption is rendered.
- On activation, `app.ui.status.get_runtime_status()` is imported and called once; the returned `{"torch": ..., "face": ...}` mapping is stored in `st.session_state["runtime_status"]` and displayed on subsequent reruns without probing again.
- The status module remains lightweight at import, and no runtime model is initialized by importing `app.main`.
- Streamlit's real `st.button` and `st.session_state` APIs are required; do not add a fallback that runs the probe automatically when a test shim omits the button API.

- [ ] **Step 1: Add a test that renders the app with the runtime button inactive and asserts `get_runtime_status` is never imported or called and no status captions appear.**
- [ ] **Step 2: Add a test that activates the button, asserts exactly one runtime probe, verifies both returned status values are displayed, and verifies a subsequent rerun uses session state without a second probe. Assert the `Check runtime` label and that the button is not shown after success.**
- [ ] **Step 3: Add a failure/retry test.** Make the first lazy status-module probe raise `RuntimeError("probe failed")`; assert the unavailable/error captions appear and the error is stored while the button remains present. On the next click, return a valid status mapping; assert the error clears, success values display, and probe count is two.
- [ ] **Step 4: Run the runtime-status tests and verify the new unloaded/error/retry assertions fail before implementation.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit/test_app_lazy_loading.py -q`

- [ ] **Step 5: Remove the top-level `get_runtime_status` import from `app/main.py`; import/call it only in the button activation branch and store the result in `st.session_state`. Require real Streamlit button/session APIs in the fake fixtures; no implicit probe fallback. Preserve exception handling so a failed probe displays unavailable/error state and can be retried.**
- [ ] **Step 6: Update the navigation fake with `session_state` and `button()` returning false by default. Update its expected call list to include the `Check runtime` button and no Torch/ONNX captions before click.**
- [ ] **Step 7: Run focused lazy-load, runtime-status, and navigation tests.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit/test_app_lazy_loading.py tests/unit/test_app_navigation.py tests/unit/test_runtime_status.py -q`

Expected: inactive tabs and status probe remain unloaded; selecting a tab or activating the button loads only the requested work.

- [ ] **Step 8: Commit on-demand runtime status.**

```powershell
git add app/main.py tests/unit/test_app_lazy_loading.py tests/unit/test_app_navigation.py
git commit -m "feat: defer runtime status probe until requested"
```

## Task 3: Disposable Runtime Validation

**Files:**
- Verify: `app/main.py`, all seven page modules, lazy-loading tests, and the current `pyproject.toml`.

- [ ] **Step 1: Run the full pytest suite once.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit tests/integration -q`

Expected: no new failures; do not change database or recognition behavior to accommodate lazy loading.

- [ ] **Step 2: Create a disposable code-only project copy including `app/` and `src/` but excluding `data/`, all legacy user-data roots, `.superpowers/`, `directmlvenv/`, and user workspace artifacts. From the copy root, set `PYTHONPATH` to both the copy root and `<copy>/src` (in that order) so `app` and `face_voice_rec` both resolve from the copy, not the live editable install. Run Streamlit from that copy.**
- [ ] **Step 3: Verify the app shows exactly the same seven tabs, no sidebar page list, no runtime status until “Check runtime” is clicked, and only the selected page is imported/rendered when stateful tabs are available. Confirm `client.showSidebarNavigation` is set to false in startup. Confirm the eager compatibility fallback with a fake older `st.tabs` API in unit tests; do not emulate an older installed Streamlit on the live workspace. Stop the disposable server after verification.**
- [ ] **Step 4: Verify the disposable copy's runtime status control populates both statuses only after click; never run the app against the live repository's `data/`.**
- [ ] **Step 5: Commit only if Task 3 required a test/validation correction; otherwise leave this task documentation-only. Do not stage live `data/`, `live/.gitkeep`, plan files, or `.superpowers/sdd/`.**

## Final Acceptance

- Stateful Streamlit tabs import and render only the selected page.
- Older Streamlit APIs preserve the current eager tab experience without raising the dependency floor.
- The Torch/ONNX runtime probe is not called until the user clicks `Check runtime`.
- Seven labels/order, page workflows, recognition thresholds, and model behavior remain unchanged.
- The database plan's one-time cached init and lock behavior remain intact.
- Tests and Streamlit smoke checks use temporary/disposable paths only.