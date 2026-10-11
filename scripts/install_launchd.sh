#!/bin/bash
# Install (or replace) the daily 05:30 local-time job. macOS only.
#   scripts/install_launchd.sh            install
#   scripts/install_launchd.sh --remove   uninstall
# launchd runs a missed job when the Mac wakes, unlike cron.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.window.daily"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if [ "${1:-}" = "--remove" ]; then rm -f "$PLIST"; echo "removed $LABEL"; exit 0; fi

mkdir -p "$HOME/Library/LaunchAgents" "$REPO/data"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$REPO/scripts/run_daily.sh</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>5</integer><key>Minute</key><integer>30</integer></dict>
  <key>StandardOutPath</key><string>$REPO/data/run.log</string>
  <key>StandardErrorPath</key><string>$REPO/data/run.log</string>
</dict>
</plist>
EOF
launchctl bootstrap "$DOMAIN" "$PLIST"
echo "installed $LABEL: daily at 05:30 (system time zone), log: $REPO/data/run.log"
