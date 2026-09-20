# LangChain backend listing readiness

Checked against the live LangChain documentation on 2026-09-20 UTC.

This package fits the backend page's stated scope: a custom virtual filesystem
that connects Deep Agents to a database. The proposed entry describes the
`GraphMemoryBackend` filesystem integration and its read-only Markdown views.
Graph mutations remain controlled Python methods and tools.

Before submitting, verify the release on
[PyPI](https://pypi.org/project/deepagents-graph-memory/) and install it in a fresh
environment. The initial audit found publication was the remaining release gate.
Maintainers decide acceptance; passing these checks cannot guarantee a merge.

## Requirements and evidence

| Check | Evidence or remaining work |
| --- | --- |
| Independent package and public source | `pyproject.toml`, MIT `LICENSE`, and the public `TahaK29/deepagents-graph-memory` repository. No implementation code belongs in the LangChain docs PR. |
| Backend protocol | Subclasses `BackendProtocol`; implements `ls`, `read`, `grep`, `glob`, `write`, `edit`, upload, and download. Writes, edits, and uploads return read-only errors. Async calls use the inherited protocol wrappers. |
| File semantics | Regression coverage checks directory paths, literal matching lines, capped search, validated relative glob patterns, newline-preserving reads and pagination, errors, and native synchronous/asynchronous `CompositeBackend` routing. |
| Bounded inspection | Directory limits return structured errors. Known node paths remain readable. Node views and graph recall retain their existing traversal budgets. |
| Runtime setup | README documents LadybugDB 0.20.3, native OpenSSL 3, one-time FTS installation, persistent storage, and a writable default backend alongside `/graph/`. |
| Supported Deep Agents versions | `>=0.6.10`, with CI checks for 0.6.10, 0.6.12, 0.7.1, and the latest release (currently 0.7.15). Native result formats and optional prompt APIs are handled across versions. Combined and graph-only agents have offline integration tests. Versions before 0.6.10 are unsupported; 0.5.2 lacks `HarnessProfile`. Future compatibility depends on passing CI. |
| Published, installable package | Verify the release's wheel and source archive on PyPI, then confirm installation from the public index in a fresh environment. |
| CI for the release revision | The latest-release matrix covers Python 3.11–3.14 on Linux, Windows, and macOS, plus three older-version jobs on Linux. Require a green run containing the final audit changes before release; earlier green runs do not validate these changes. |

The required filesystem methods come from the
[custom backend guide](https://docs.langchain.com/oss/python/deepagents/backends#custom-backends).
The [integration contribution guide](https://docs.langchain.com/oss/python/contributing/integrations-langchain)
requires independently published packages. Its standard-test requirement says
"if applicable"; the backend contract is covered directly here rather than by a
chat-model or vector-store test suite.

## Submission route

The [backend index](https://docs.langchain.com/oss/python/integrations/backends)
explicitly invites a PR adding a table row. The target source file is
[`src/oss/integrations/backends/index.mdx`](https://github.com/langchain-ai/docs/blob/main/src/oss/integrations/backends/index.mdx).
Keep this change to one row linking to the package README.

There is conflicting general guidance: the
[publishing guide](https://docs.langchain.com/oss/python/contributing/publish-langchain#make-your-integration-discoverable)
asks for an Integration listing issue and says not to open a manual listing PR
unless a maintainer requests it. Its current
[issue form](https://github.com/langchain-ai/docs/blob/main/.github/ISSUE_TEMPLATE/06-integration-submission.yml)
has no `backends` component. After publication, confirm the backend-specific route
with a maintainer if that discrepancy remains. Do not select `graphs` or `sandboxes`
just to fit the form: this entry is a filesystem backend.

The 50,000-monthly-download threshold governs a new hosted integration guide,
not a claim of eligibility for this existing backend table. This submission
requests no new guide, navigation entry, or featured status.

## Proposed table row

```markdown
| [Graph Memory Backend](https://github.com/TahaK29/deepagents-graph-memory#quick-start) | Read-only filesystem backend that exposes LadybugDB project context as Markdown, with controlled graph tools for updates. | `deepagents-graph-memory` | [`TahaK29/deepagents-graph-memory`](https://github.com/TahaK29/deepagents-graph-memory) |
```

Suggested title: `docs: list Graph Memory Backend for Deep Agents`

Use the upstream PR template when the submission route is confirmed. The overview
can read:

> Add Graph Memory Backend to the existing backend table. The independently
> maintained package exposes project entities and workflow traces as read-only
> Markdown through Deep Agents' filesystem tools. Graph updates use controlled
> tools, and the README documents installation, native prerequisites, supported
> versions, and CompositeBackend setup.

State that Codex assisted with the audit and draft, as required by the docs
repository's contribution instructions. Attach links to the published package
and the final green CI run. Do not check the template's `docs dev` box until that
preview has actually run.

## Before publication

Run these from the package root after installing test and build tools and
provisioning FTS as described in the README:

```bash
python -m pytest -q
python -m ruff check .
python -m build
python -m twine check dist/*
```

Test the built wheel in a fresh environment, including graph search and reopening
a persistent database. Publish only the tested artifacts under the maintainers'
PyPI account.

## Publishing a release

The manually triggered `.github/workflows/publish.yml` workflow publishes from
`main` after the Tests workflow succeeds for that exact commit. It builds and
validates both distributions, then uploads them using PyPI Trusted Publishing.
Only the upload job has permission to request a publishing identity.

Configure the PyPI publisher with project `deepagents-graph-memory`, owner
`TahaK29`, repository `deepagents-graph-memory`, workflow `publish.yml`, and
environment `pypi`. For the first release, add this as a pending publisher on the
maintainer's PyPI account. No long-lived API token is needed.

After updating the package version and waiting for its CI run, start the workflow:

```bash
gh workflow run publish.yml --ref main
```

Confirm the uploaded version and a clean public-index installation before
announcing the release or submitting the LangChain listing.
