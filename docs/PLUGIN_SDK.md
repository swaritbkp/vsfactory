# Plugin SDK: shipping a task domain as an external wheel

A plugin samples a scene from a seed, renders it, and declares its claims and its answer. It is
**untrusted**. The Knotwork kernel re-derives every answer from renderer object masks, cross-checks
it against the visible image, and quarantines the item when the plugin is wrong. A plugin can
therefore be written, versioned and installed separately, without review as part of the trusted
kernel. A plugin that renders or labels wrongly lowers its own yield rather than corrupting labels.
Plugin code still runs as ordinary Python in the generation process. Certification protects label
validity; it does not sandbox the host. Install only plugins you trust to execute.

## 1. Declare the entry point

```toml
# pyproject.toml of your package
[project]
name = "knotwork-dials"
version = "0.1.0"
dependencies = ["knotwork"]

[project.entry-points."knotwork.plugins"]
dials = "knotwork_dials.plugin:PLUGIN"   # or "knotwork_dials" if the module exports PLUGIN
```

The entry-point name must equal `PLUGIN.name`. It must not collide with a built-in plugin
(`rack_patch`, `tangle`, `gauge`, `panel`) or with another registration.

## 2. Implement the protocol

`PLUGIN` must satisfy `knotwork.knotcore.contracts.TaskPlugin`:

| Member | Purpose |
|---|---|
| `name`, `version`, `families` | Identity and the question families the plugin can ask |
| `sample_scene(seeds, profile, difficulty=None)` | A seeded `SceneIR`; all randomness comes from `seeds` |
| `render(scene, out_dir, fault=None)` | Beauty image 2× supersampled and downsampled to 1024 px; amodal masks and the modal 16-bit id buffer at 1× with no anti-aliasing |
| `claims(scene)` | Declarative claims the kernel cross-checks against the pixels |
| `questions(scene, seeds, family)` | A `QuestionProgram` the kernel executes |
| `symbolic_answer(scene, program)` / `geometric_answer(scene, program)` | The plugin's own answers (D1 / D2) |
| `causal_twin(...)` / `nuisance_twin(...)` | Counterfactual twins: the answer changes / the answer stays |

Import only `knotwork.knotcore` (contracts, seeds) and `knotwork.render`. The fastest start is
`uv run knot new-plugin dials --dest src/`. It writes a working skeleton with its conformance tests
and adds no lines to `knotcore`.

## 3. Install and check

```bash
uv pip install -e ./knotwork-dials     # or: uv pip install dist/knotwork_dials-0.1.0-py3-none-any.whl
uv run knot plugins                    # status must be "ok"; a rejection prints the reason
uv run knot plugins --json             # name, source, version, status, origin, detail
```

Checklist:

- [ ] `knot plugins` lists the plugin with `source = external` and `status = ok`.
- [ ] The conformance test from `knot new-plugin` passes: clean scenes certify.
- [ ] Injected faults from `knot mutate --fault all` are quarantined on your renders.
- [ ] Rendering twice from one seed gives identical files: no `random`, `hash()` or wall-clock time.
- [ ] Nothing touches the network at import or render time (`KNOTWORK_NETWORK=deny` is the default).

The registry rejects a registration, and stays usable, when the plugin fails to import, lacks a
protocol method or attribute, collides with a registered name, or has an entry-point name that
differs from `PLUGIN.name`.
