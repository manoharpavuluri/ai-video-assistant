#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="AI Video Assistant"
APP_DIR="$ROOT_DIR/dist/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
DMG_PATH="$ROOT_DIR/dist/$APP_NAME.dmg"
STAGING_DIR="$ROOT_DIR/dist/dmg-staging"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing virtualenv at $ROOT_DIR/.venv. Create it and install requirements first."
  exit 1
fi

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$STAGING_DIR"
cp "$ROOT_DIR/assets/app_icon.icns" "$RESOURCES_DIR/app_icon.icns"

cat > "$CONTENTS_DIR/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>ai-video-assistant</string>
  <key>CFBundleIdentifier</key>
  <string>local.ai-video-assistant</string>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundleDisplayName</key>
  <string>$APP_NAME</string>
  <key>CFBundleIconFile</key>
  <string>app_icon</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

cat > "$MACOS_DIR/ai-video-assistant" <<LAUNCHER
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$ROOT_DIR"
cd "\$ROOT_DIR"

export PATH="\$ROOT_DIR/.venv/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

PORT="\${AI_VIDEO_ASSISTANT_PORT:-8501}"
URL="http://localhost:\$PORT"

"\$ROOT_DIR/.venv/bin/python" -m streamlit run "\$ROOT_DIR/streamlit_app.py" \\
  --server.port "\$PORT" \\
  --server.headless true \\
  --server.fileWatcherType none \\
  --browser.gatherUsageStats false &

streamlit_pid="\$!"

for _ in {1..60}; do
  if curl -fsS "\$URL" >/dev/null 2>&1; then
    open "\$URL" >/dev/null 2>&1 || true
    break
  fi
  sleep 0.5
done

wait "\$streamlit_pid"
LAUNCHER

chmod +x "$MACOS_DIR/ai-video-assistant"

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
cp -R "$APP_DIR" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

rm -f "$DMG_PATH"
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

echo "$DMG_PATH"
