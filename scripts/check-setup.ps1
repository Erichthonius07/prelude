# Preflight check for Prelude dev environment. Report-only — fixes nothing automatically.
# Keep in sync with check-setup.sh — same checks, same order.

$failures = 0

function Write-Check($name, $status, $detail) {
    $color = switch ($status) { "PASS" {"Green"} "WARN" {"Yellow"} "FAIL" {"Red"} }
    Write-Host ("[{0}] {1}" -f $status, $name) -ForegroundColor $color
    if ($detail) { Write-Host "    $detail" -ForegroundColor Gray }
}

$dockerVersion = docker --version 2>$null
if (-not $dockerVersion) {
    Write-Check "Docker" "FAIL" "Not found. Install Docker Desktop."
    $failures++
} else {
    docker info *>$null
    if ($LASTEXITCODE -eq 0) { Write-Check "Docker" "PASS" $dockerVersion }
    else { Write-Check "Docker" "FAIL" "Installed but daemon not running."; $failures++ }
}

$composeVersion = docker compose version 2>$null
if ($composeVersion) { Write-Check "Docker Compose" "PASS" $composeVersion }
else { Write-Check "Docker Compose" "FAIL" "Not available. Update Docker Desktop."; $failures++ }

$javaOut = java -version 2>&1 | Out-String
if ($javaOut -match '"(\d+)') {
    if ($matches[1] -eq "25") { Write-Check "JDK" "PASS" "Java $($matches[1])" }
    else { Write-Check "JDK" "WARN" "Java $($matches[1]) found, project expects 25."; $failures++ }
} else {
    Write-Check "JDK" "FAIL" "java not found on PATH."; $failures++
}

$sdk = $env:ANDROID_HOME
if (-not $sdk) { $sdk = $env:ANDROID_SDK_ROOT }
if ($sdk -and (Test-Path $sdk)) { Write-Check "Android SDK" "PASS" "Found at $sdk" }
else { Write-Check "Android SDK" "FAIL" "ANDROID_HOME/ANDROID_SDK_ROOT not set. Open project in Android Studio once."; $failures++ }

if (Test-Path "capture-android\gradlew.bat") { Write-Check "Gradle wrapper" "PASS" "present" }
else { Write-Check "Gradle wrapper" "FAIL" "capture-android\gradlew.bat missing."; $failures++ }

if (Test-Path ".env") { Write-Check ".env" "PASS" "present" }
else { Write-Check ".env" "FAIL" "Missing. Run: Copy-Item .env.example .env"; $failures++ }

Write-Host ""
if ($failures -eq 0) { Write-Host "All checks passed." -ForegroundColor Green }
else { Write-Host "$failures check(s) failed." -ForegroundColor Red }
