# Releasing

Releases are published by `.github/workflows/release.yml` using PyPI
**Trusted Publishing** (OIDC). There is no API token in repo secrets — PyPI
mints a short-lived credential scoped to this repository and workflow, so
there is nothing long-lived to leak or rotate.

## One-time setup (before the first release)

Trusted Publishing has to be configured on PyPI's side, and for a project
that does not exist yet this is done as a **pending publisher**:

1. Sign in to PyPI → *Your projects* → *Publishing* → *Add a pending publisher*.
2. Fill in exactly:
   - PyPI project name: `inferenceswitch`
   - Owner: `IDEMSInternational`
   - Repository: `inferenceswitch`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. Repeat on [test.pypi.org](https://test.pypi.org) with environment `testpypi`
   if you want the rehearsal path.
4. In this repo's *Settings → Environments*, create `pypi` (and `testpypi`).
   Adding a required reviewer to `pypi` makes every upload need a human
   click, which is worth it — an upload cannot be undone.

The names must match character for character. A mismatch fails at upload with
an unhelpful permissions error, which is the usual reason a first release
fails.

## Rehearsing

*Actions → Release → Run workflow* publishes to **TestPyPI** without touching
PyPI. Safe to repeat: that job sets `skip-existing`, so re-running the same
version is not an error.

Install the result with:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ inferenceswitch
```

The extra index is needed because TestPyPI does not carry the provider SDKs.

## Releasing for real

1. Bump `version` in `pyproject.toml` and move the `[Unreleased]` entries in
   `CHANGELOG.md` under the new number.
2. Merge that to `main`.
3. Create a GitHub Release with tag `v<version>` — e.g. `v0.1.0`.

Publishing the release triggers the workflow. It verifies the tag matches
`pyproject.toml`, builds an sdist and a wheel, runs `twine check --strict`,
and uploads.

**Version numbers are not reusable.** PyPI refuses to replace an existing
file, and yanking hides a release without freeing its filename. That is why
the workflow checks tag against `pyproject.toml` before building rather than
discovering the mismatch at upload — a wrong number cannot be corrected, only
superseded.

## After the first release

`IDEMSInternational/svc` currently depends on this package by git URL,
because it was not on PyPI:

```toml
"inferenceswitch[anthropic,gemini] @ git+https://github.com/IDEMSInternational/inferenceswitch.git@main"
```

Once a release exists that becomes an ordinary specifier
(`"inferenceswitch[anthropic,gemini]>=0.1.0"`), which is worth doing — a git
reference always resolves to a moving branch rather than a fixed version.
