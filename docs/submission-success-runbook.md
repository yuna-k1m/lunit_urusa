# Successful CoEval Submission Runbook

This note records how the team produced a successful Lunit hackathon submission and how to avoid
the failure that delayed it. The successful release branch was `lunit/hackathon-submission`; the
final verified release SHA was:

```text
d2401b8fa981cdcec3ecb4956996aeba28d546bf
```

## What the evaluator required

The submitted repository had to build a container in under five minutes, start without manual
setup, listen on `0.0.0.0:8000`, and implement these OpenAI-compatible endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`

The evaluator sends the complete conversation history in `messages`. The driver forwards that
history to `Lunit/L2-preview` and returns the upstream OpenAI-compatible response.

## The failure sequence

The first model proxy expected `LUNIT_FM_API_KEY` to be injected at container runtime. CoEval did
not inject it, so chat requests could not authenticate. We temporarily replaced the proxy with an
offline static response to prove that the image itself could build, start, and serve the required
API.

When the model proxy was restored, the credential fallback was not restored with it. The service
therefore returned HTTP 503 for every chat request with `Server is missing LUNIT_FM_API_KEY.` The
pipeline surfaced only a final `docker start --attach` nonzero exit, which was a misleading wrapper
error rather than the actionable application error.

The decisive diagnostic was to rebuild the exact release for `linux/amd64`, run it without any
environment variables, and call both endpoints. Uvicorn stayed alive, `/v1/models` returned 200,
and `/v1/chat/completions` returned 503. That isolated credential availability as the regression.

## The working configuration

The successful submission keeps `submission_api_key` as a tracked file and copies it into the
image:

```dockerfile
COPY app.py submission_api_key ./
```

At runtime, the application resolves the credential in this order:

1. Use `LUNIT_FM_API_KEY` when it is present.
2. Otherwise read `/app/submission_api_key`.
3. Return a clear server error if neither source is available.

This preserves local and managed-secret overrides while allowing CoEval to run without injecting a
runtime secret. `submission_api_key` is part of the submission contract for this repository: do
not delete it or remove it from the Docker `COPY` instruction.

Because a tracked credential is visible in Git history and the built image, use a dedicated,
short-lived hackathon key and rotate it after evaluation. Never include its value in documentation,
logs, test output, or model responses.

## Reproducible validation

Build for the evaluator architecture:

```bash
docker build --platform linux/amd64 -t urusa-submission:local .
```

Run the tests inside that image:

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD/tests:/app/tests:ro" \
  urusa-submission:local \
  python -m unittest discover -s tests -v
```

Confirm that the bundled credential is available without printing it:

```bash
docker run --rm --platform linux/amd64 urusa-submission:local \
  python -c 'from app import get_api_key; assert get_api_key()'
```

Start the service exactly as CoEval will:

```bash
docker run --rm --platform linux/amd64 -p 8000:8000 urusa-submission:local
```

From another shell, verify the API contract:

```bash
curl --fail http://127.0.0.1:8000/v1/models

curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Lunit/L2-preview",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

The second request requires access to the Lunit network. Outside that network, validate credential
loading and mock the upstream response rather than treating network unavailability as a container
startup failure.

## Release procedure

1. Start a focused branch from the latest `lunit/hackathon-submission`.
2. Run the `linux/amd64` build and contract tests above.
3. Verify that `submission_api_key` exists in the Git tree and is non-empty without displaying it.
4. Verify that the Dockerfile copies the key and the application retains the file fallback.
5. Open and merge a pull request into `lunit/hackathon-submission`.
6. Fetch the remote branch and obtain its full SHA:

   ```bash
   git fetch origin lunit/hackathon-submission
   git rev-parse origin/lunit/hackathon-submission
   ```

7. Submit that exact 40-character SHA to the hackathon dashboard.
8. After a successful run, record the SHA and delete merged feature branches.

The central lesson is to validate the submitted image under the evaluator's actual assumptions:
correct architecture, no injected environment variables, required files present in the image, and
the service reachable on port 8000.
