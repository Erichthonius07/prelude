#!/usr/bin/env bash
# Preflight check for Prelude dev environment. Report-only — fixes nothing automatically.
# Keep in sync with check-setup.ps1 — same checks, same order.

failures=0
pass() { echo -e "\033[32m[PASS]\033[0m $1 ${2:+- $2}"; }
warn() { echo -e "\033[33m[WARN]\033[0m $1 ${2:+- $2}"; failures=$((failures+1)); }
fail() { echo -e "\033[31m[FAIL]\033[0m $1 ${2:+- $2}"; failures=$((failures+1)); }

if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then pass "Docker" "$(docker --version)"
    else fail "Docker" "Installed but daemon not running."; fi
else fail "Docker" "Not found."; fi

if docker compose version >/dev/null 2>&1; then pass "Docker Compose" "$(docker compose version)"
else fail "Docker Compose" "Not available."; fi

if command -v java >/dev/null 2>&1; then
    ver=$(java -version 2>&1 | head -1)
    if echo "$ver" | grep -q '"25'; then pass "JDK" "$ver"
    else warn "JDK" "$ver — project expects JDK 25."; fi
else fail "JDK" "java not found on PATH."; fi

sdk_path="${ANDROID_HOME:-$ANDROID_SDK_ROOT}"
if [ -n "$sdk_path" ] && [ -d "$sdk_path" ]; then pass "Android SDK" "Found at $sdk_path"
else fail "Android SDK" "Not set or missing. Open project in Android Studio once."; fi

if [ -f "capture-android/gradlew" ] && [ -x "capture-android/gradlew" ]; then
    pass "Gradle wrapper" "present and executable"
else fail "Gradle wrapper" "missing or not executable."; fi

if [ -f ".env" ]; then pass ".env" "present"
else fail ".env" "Missing. Run: cp .env.example .env"; fi

echo ""
if [ "$failures" -eq 0 ]; then echo "All checks passed."
else echo "$failures check(s) failed."; fi
