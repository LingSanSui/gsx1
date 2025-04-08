#!/bin/bash
set -euo pipefail

# Downloads a given artifact from the latest green build of a certain pipeline on its main branch.
# We can get fancier parameterizing this later.
# Requires `BUILDKITE_API_ACCESS_TOKEN` be set. If you wish to run this locally,
# make one for your Buildkite User.
#
# Usage: download-buildkite-artifact.sh <pipeline> <artifact>

# This graphQL query ought to return the latest green build in the other repo.
# It _should_ sort by the latest, just keep an eye on it.
response=$(curl -s https://graphql.buildkite.com/v1 \
  -H "Authorization: Bearer $BUILDKITE_API_ACCESS_TOKEN" \
  -d @- <<EOGRAPHQL
{
    "query": "{ pipeline(slug: \"gensyn-ai/$1\") { builds(branch: \"main\" state: PASSED first: 1) { edges { node { uuid } } } } }"
}

EOGRAPHQL
)

build_id=$(echo "$response" | jq -r '.data.pipeline.builds.edges[0].node.uuid')

buildkite-agent artifact download --build "$build_id" "$2" .
