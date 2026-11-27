# Deploying the Studio as a Cloud Run sandbox

One container image serves two deployments. The switch is `KNOTWORK_PUBLIC_DEMO`.

| | Self-hosted Docker (default) | Cloud Run sandbox |
|---|---|---|
| Switch | `KNOTWORK_PUBLIC_DEMO` unset or `0` | `KNOTWORK_PUBLIC_DEMO=1` |
| Storage | any folder you mount | `/tmp/knotwork_demo/` only (in-memory, lost on restart) |
| Scale | up to 100,000 items per block | 10 items per block, 3 blocks, *k* ≤ 3, 10-item Studio batches |
| Model Arena | Offline Mock or Live (OpenRouter) | Offline Mock only: no Live mode, no API key field |
| API keys | `.env`, `-e OPENROUTER_API_KEY`, or pasted for one session | never read, never accepted |
| `campaign.yaml` import | available | hidden |
| Plugins | built-ins plus `knotwork.plugins` entry points from installed wheels | the 4 built-in plugins only |
| Network | `KNOTWORK_NETWORK=deny`; live runs are an explicit opt-in | the same |

## What the sandbox enforces

`knotwork.studio.demo_guard` is active only when `KNOTWORK_PUBLIC_DEMO` is exactly `1`. When active:

- Every folder a visitor can type (Studio batch output, Arena `data_dir` / `results_dir` / register
  folder, Export output) is resolved, with symlinks and `..` expanded, and refused unless it lies
  inside `/tmp/knotwork_demo/`.
- Studio batch size, items per block, block count, repeats and spend are capped server-side. The
  caps hold whatever the UI submits.
- Model Arena runs **Offline Mock only**. The `Live (OpenRouter)` mode and every API key field are
  removed from the page, and the launch handler (`knotwork.arena.planning.start`) forces
  `mode="mock"` with no key whatever the request says. Resuming a live session is refused. The
  server's `OPENROUTER_API_KEY` and `.env` are never read.
- The `Import campaign.yaml` uploader is hidden.
- External plugins are not loaded: the registry skips the `knotwork.plugins` entry-point scan, so
  only `tangle`, `gauge`, `panel` and `rack_patch` run.
- A sandbox banner is shown on all five pages: "Live Interactive Sandbox (Offline Mock only ·
  capped at 10 items/block · ephemeral /tmp storage)".

With the variable unset or `0`, the guard changes nothing. `tests/apps/test_demo_guard.py` covers
confinement, traversal, caps, key isolation, the Offline-Mock-only lock (UI and server side), the
hidden campaign import, the skipped entry-point scan and the no-op case.

The sandbox is for demonstrating the system. Run real campaigns locally or in self-hosted Docker.

## 1. Deploy

From the repository root, with the `gcloud` CLI signed in and a project selected.

Bash:

```bash
gcloud run deploy knotwork-demo \
  --source . \
  --region asia-south1 \
  --memory=2Gi --cpu=2 \
  --timeout=3600 \
  --session-affinity \
  --min-instances=0 --max-instances=2 \
  --set-env-vars KNOTWORK_PUBLIC_DEMO=1,KNOTWORK_NETWORK=deny \
  --allow-unauthenticated
```

PowerShell (backtick continues a line; the comma-separated value must be quoted):

```powershell
gcloud run deploy knotwork-demo `
  --source . `
  --region asia-south1 `
  --memory=2Gi --cpu=2 `
  --timeout=3600 `
  --session-affinity `
  --min-instances=0 --max-instances=2 `
  --set-env-vars "KNOTWORK_PUBLIC_DEMO=1,KNOTWORK_NETWORK=deny" `
  --allow-unauthenticated
```

`--source .` builds the repository `Dockerfile` with Cloud Build. Cloud Run sets `PORT`, and the
image's default command serves the Studio on `${PORT:-8501}`. Streamlit holds one WebSocket per
visitor, so the long `--timeout` and `--session-affinity` keep a session on one instance.

To build and push the image yourself (replace `PROJECT_ID`):

```bash
docker build -t asia-south1-docker.pkg.dev/PROJECT_ID/knotwork/knotwork:v2 .
docker push asia-south1-docker.pkg.dev/PROJECT_ID/knotwork/knotwork:v2
gcloud run deploy knotwork-demo --image asia-south1-docker.pkg.dev/PROJECT_ID/knotwork/knotwork:v2 \
  --region asia-south1 --memory=2Gi --cpu=2 --timeout=3600 --session-affinity \
  --min-instances=0 --max-instances=2 \
  --set-env-vars KNOTWORK_PUBLIC_DEMO=1,KNOTWORK_NETWORK=deny --allow-unauthenticated
```

Do not pass `OPENROUTER_API_KEY` to the service. The sandbox ignores it.

## 2. Smoke check

Bash:

```bash
URL=$(gcloud run services describe knotwork-demo --region asia-south1 --format 'value(status.url)')
curl -fsS "$URL/_stcore/health"          # prints "ok"
```

PowerShell:

```powershell
$URL = gcloud run services describe knotwork-demo --region asia-south1 --format "value(status.url)"
Invoke-RestMethod "$URL/_stcore/health"   # prints "ok"
```

Then open the URL in a browser and check four things:

1. The sandbox banner shows on all five pages.
2. Studio: a 10-item batch builds, and sample count, batch size and repeats stop at their caps.
3. Model Arena: the default Offline Mock campaign launches and finishes. A results folder outside
   `/tmp/knotwork_demo/` is refused with a message.
4. Model Arena offers only Offline Mock: no Live mode, no API key field and no
   `Import campaign.yaml` uploader.

The same check runs locally against the image:

```bash
docker run --rm -p 8080:8080 -e PORT=8080 -e KNOTWORK_PUBLIC_DEMO=1 knotwork:v2
```

## 3. Cost and teardown

With `--min-instances=0` the service scales to zero, so an idle sandbox costs only image storage.
While in use, one instance with 2 vCPU and 2 GiB is billed per second, and `--max-instances=2`
bounds the worst case. Check current Cloud Run prices before a public event.

Set a billing budget alert before you publish the URL. `--max-instances=2` caps concurrency, not
spend over time. In the Google Cloud console open **Billing → Budgets & alerts → Create budget**,
scope it to the project, set a monthly amount (for example $10) and keep the default 50 %, 90 % and
100 % email thresholds. The same from the CLI (replace `BILLING_ACCOUNT_ID` and `PROJECT_ID`):

```bash
gcloud billing budgets create --billing-account=BILLING_ACCOUNT_ID \
  --display-name="knotwork-demo" --budget-amount=10USD \
  --filter-projects=projects/PROJECT_ID \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
```

A budget alert notifies you; it does not stop the service. Delete the service (below) to stop billing.

```bash
gcloud run services delete knotwork-demo --region asia-south1
gcloud artifacts docker images list asia-south1-docker.pkg.dev/PROJECT_ID/cloud-run-source-deploy   # then delete old images
```
