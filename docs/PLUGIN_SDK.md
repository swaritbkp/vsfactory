# Knotwork Plugin SDK — ship a task domain as an external wheel

A plugin samples a scene, renders it, and declares claims and an answer. It is **untrusted**: the
Knotwork kernel re-derives every answer from renderer object masks, cross-checked against the
visible image, and quarantines the item when the plugin is wrong. So a plugin can be written and
installed separately without being reviewed as part of the trusted kernel.

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

The entry point name must equal `PLUGIN.name` and must not collide with a built-in plugin
(`rack_patch`, `tangle`, `gauge`, `panel`) or with another registration.

## 2. Implement the protocol

`PLUGIN` must satisfy `knotwork.knotcore.contracts.TaskPlugin`:

| Member | Purpose |
|---|---|
| `name`, `version`, `families` | identity and the question families it can ask |
| `sample_scene(seeds, profile, difficulty=None)` | a seeded `SceneIR` (all randomness from `seeds`) |
| `render(scene, out_dir, fault=None)` | beauty (2× supersampled), amodal masks and modal id buffer at 1×, no anti-aliasing |
| `claims(scene)` | declarative claims the kernel cross-checks |
| `questions(scene, seeds, family)` | a `QuestionProgram` |
| `symbolic_answer(scene, program)` / `geometric_answer(scene, program)` | the plugin's D1 / D2 answers |
| `causal_twin(...)` / `nuisance_twin(...)` | counterfactual twins (answer changes / stays) |

Import only `knotwork.knotcore` (contracts, seeds) and `knotwork.render`. The fastest start is
`uv run knot new-plugin dials --dest src/`, which writes a working skeleton and its tests.

## 3. Verify

```bash
pip install -e ./knotwork-dials        # or: uv pip install dist/knotwork_dials-0.1.0-py3-none-any.whl
uv run knot plugins                    # status must be "ok"; a rejection prints the reason
uv run knot plugins --json             # name, source, version, status, origin, detail
```

Checklist:

- [ ] `knot plugins` lists it with `source = external` and `status = ok`.
- [ ] Your conformance test (from `knot new-plugin`) passes: clean scenes certify.
- [ ] `knot mutate --fault all` injected faults are caught (quarantined) on your renders.
- [ ] No `random`, `hash()` or wall-clock time: rendering twice from one seed gives identical files.
- [ ] No network access at import or render time (`KNOTWORK_NETWORK=deny` is the default).

A registration is **rejected** (never crashing the registry) when it fails to import, misses a
protocol method or attribute, collides with a registered name, or its entry point name differs from
`PLUGIN.name`.
