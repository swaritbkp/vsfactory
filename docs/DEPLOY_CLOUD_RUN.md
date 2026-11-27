# Deploy the Studio as a Cloud Run sandbox demo

Knotwork runs two ways from the same image:

| | Self-hosted Docker (default) | Cloud Run public demo |
|---|---|---|
| Switch | `KNOTWORK_PUBLIC_DEMO` unset or `0` | `KNOTWORK_PUBLIC_DEMO=1` |
| Storage | any folder you mount | `/tmp/knotwork_demo/` only (in-memory, lost on restart) |
| Scale | up to 100,000 items per block | 10 items per block, 3 blocks, k ≤ 3, 10-item Studio batches |
| API keys | `.env`, `-e OPENROUTER_API_KEY` or in-session | never read from the server; visitors may paste their own key for one browser session |
| Network | `KNOTWORK_NETWORK=deny`; live runs are an explicit opt-in | the same |

The demo is for showing the factory. Run real campaigns on your own machine or in Docker.

## 1. Deploy

From the repository root, with the `gcloud` CLI signed in and a project selected:

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

`--source .` builds the repository `Dockerfile` with Cloud Build. Cloud Run sets `PORT`, and the
image's default command serves the Studio on `${PORT:-8501}`. Streamlit keeps a WebSocket open per
visitor, so the long `--timeout` and `--session-affinity` keep a session on one instance.

To build the container yourself instead:

```bash
docker build -t asia-south1-docker.pkg.dev/PROJECT_ID/knotwork/knotwork:v2 .
docker push asia-south1-docker.pkg.dev/PROJECT_ID/knotwork/knotwork:v2
gcloud run deploy knotwork-demo --image asia-south1-docker.pkg.dev/PROJECT_ID/knotwork/knotwork:v2 \
  --region asia-south1 --memory=2Gi --cpu=2 --timeout=3600 --session-affinity \
  --min-instances=0 --max-instances=2 \
  --set-env-vars KNOTWORK_PUBLIC_DEMO=1,KNOTWORK_NETWORK=deny --allow-unauthenticated
```

Never pass `OPENROUTER_API_KEY` to the service: the demo ignores it anyway.

## 2. Smoke-check

```bash
URL=$(gcloud run services describe knotwork-demo --region asia-south1 --format 'value(status.url)')
curl -fsS "$URL/_stcore/health"          # prints "ok"
```

Then open `$URL` in a browser and check:

1. The blue sandbox banner shows on all five pages.
2. Studio: a batch of 10 items builds. Sample count, batch size and repeats stop at their caps.
3. Model Arena: the default Offline Mock campaign launches and finishes. A results folder outside
   `/tmp/knotwork_demo/` is refused with a message.
4. Live mode asks for your own key and shows no "key loaded" badge.

Locally, the same check is `docker run --rm -p 8080:8080 -e PORT=8080 -e KNOTWORK_PUBLIC_DEMO=1 knotwork:v2`.

## 3. Cost and teardown

With `--min-instances=0` the service scales to zero, so an idle demo costs nothing beyond stored
images. While someone uses it, one instance with 2 vCPU and 2 GiB is billed per second of use, and
`--max-instances=2` bounds the worst case. Check current prices on the Cloud Run pricing page before
a public event.

```bash
gcloud run services delete knotwork-demo --region asia-south1
gcloud artifacts docker images list asia-south1-docker.pkg.dev/PROJECT_ID/cloud-run-source-deploy  # then delete old images
```
