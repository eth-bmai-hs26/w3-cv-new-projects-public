#!/usr/bin/env bash
# Double-click in Finder to start the tile inspection platform on macOS.
# (First time: if macOS says it cannot verify the file, right-click it and choose Open.)
cd "$(dirname "$0")"
./run.sh "$@" || { echo; echo "The platform stopped with an error - see the messages above."; read -r -p "Press Enter to close. "; }
