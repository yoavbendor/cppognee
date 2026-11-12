> **Status as of 2026-11-12 (this fork).** Workflows in this repository no longer publish
> outside GitHub, and no AI review, triage or summary service receives repository content.
> Test suites still call LLM providers to test cognee's own integration. The Docker Hub
> and PyPI publishing jobs and the Mintlify and community dispatches were removed from
> `release.yml`, and the Docker Hub, dev-canary and MCP PyPI workflows were deleted. The
> release notes are GitHub's generated notes; the AI-written notes step was removed. A
> release now creates the tag and the GitHub release, and nothing else. The steps below
> keep the upstream wording; the Docker Hub and PyPI checks in "Release validation" do not
> apply to this fork.

## How to make a release

### Dev release

#### Prepare release
1. Set the project version that will be released in [pyproject.toml](https://github.com/topoteretes/cognee/blob/dev/pyproject.toml#L4)
2. Update `uv.lock` with `uv lock` lock command
3. Create a PR with the changes mentioned above to `dev` and merge it.

#### Perform Release
1. Go to [Release action](https://github.com/topoteretes/cognee/actions/workflows/release.yml)
2. Select `dev` branch and run the workflow.
3. Watch the logs and make sure that everything goes well

### Main release

#### Prepare release
1. Set the project version that will be released in [pyproject.toml](https://github.com/topoteretes/cognee/blob/dev/pyproject.toml#L4)
2. Update `uv.lock` with `uv lock` lock command
3. Create a PR with the changes mentioned above to `main` and merge it.

#### Perform Release
1. Go to [Release action](https://github.com/topoteretes/cognee/actions/workflows/release.yml)
2. Select `main` branch and run the workflow.
3. Watch the logs and make sure that everything goes well

### Release validation

As of 2026-11-12 (this fork), only step 3 applies. Steps 1 and 2 describe upstream publishing that these workflows no longer perform.

1. ~~Make sure that the correct image is published to Docker Hub~~ (not performed in this fork since 2026-11-12)
2. ~~Python package is published to PyPI~~ (not performed in this fork since 2026-11-12)
3. Find the created github release in [GitHub releases](https://github.com/topoteretes/cognee/releases). Edit the release notes if required; they are GitHub's generated notes.
