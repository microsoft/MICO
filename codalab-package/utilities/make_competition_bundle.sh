#!/bin/bash
# Usage:
#   make_competition_bundle.sh <competition data path> <codalab package template> scoring program path> (<name>)
#
set -e
set -u
set -x

COMPETITION_DATA_PATH=$(realpath "$1")
CODALAB_PACKAGE_TEMPLATE=$(realpath "$2")
BUNDLE_NAME=bundle-$(date "+%Y-%m-%d-%H-%M-%S")-"$(basename $COMPETITION_DATA_PATH)"
SCORING_PROGRAM_DIR=$(realpath "$3")

tmp="$PWD/$BUNDLE_NAME.tmp/"

# Create archives for the reference data (solutions)
mkdir -p "$tmp/reference_data_dev/$(basename $COMPETITION_DATA_PATH)"
mkdir -p "$tmp/reference_data_final/$(basename $COMPETITION_DATA_PATH)"

# NOTE: We need to to be in $COMPETITION_DATA_PATH for the following to work.
cd "$COMPETITION_DATA_PATH/"
find . -name solution.csv | grep dev | xargs -i cp --parents {} "$tmp/reference_data_dev/$(basename $COMPETITION_DATA_PATH)"
find . -name solution.csv | grep final | xargs -i cp --parents {} "$tmp/reference_data_final/$(basename $COMPETITION_DATA_PATH)"

cd "$tmp"

# Convert Markdown to HTML pages
for fname in "$CODALAB_PACKAGE_TEMPLATE"/*.md; do
  echo Converting $fname to HTML
  pandoc "$fname" -o ./$(basename "${fname%.*}").html
done

# Tweak and copy competition YAML file
if [ -z "$4" ]
  then
    cp "$CODALAB_PACKAGE_TEMPLATE/competition.yaml" .
else
    CHANGES="(.title = \"MICO - $4\")"
    if [[ "$4" == "DP Distinguisher" ]]
      then
        CHANGES="$CHANGES|\
(.leaderboard.leaderboards.Results_1.label = \"CIFAR-10\")|\
(.leaderboard.leaderboards.Results_2.label = \"Purchase-100\")|\
(.leaderboard.leaderboards.Results_3.label = \"SST-2\")"
    fi
    cat "$CODALAB_PACKAGE_TEMPLATE/competition.yaml" | yq e "$CHANGES" > competition.yaml
fi

# Copy logo and scoring program
cp "$CODALAB_PACKAGE_TEMPLATE/logo.png" .
cp -r "$SCORING_PROGRAM_DIR" scoring_program

# Zip individual bundle components
for dir in reference_data_dev reference_data_final scoring_program; do
  echo Zipping $dir
  # Zip without including main directory
  cd $dir
  zip -r ../$dir.zip .
  cd ..
  rm -rf $dir
done

# Zip the bundle
zip -r ../$BUNDLE_NAME.zip *
cd ..
rm -rf "$tmp"
