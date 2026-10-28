<#
scripts/ingest_from_windows.ps1 — one command that gets the workbook into Postgres.

WHY THIS EXISTS. The team-game ingest is the one step in this project that
cannot be automated: the BigDataBall export is a licensed file on an operator's
machine, and Railway's Postgres is only reachable from there through its TCP
proxy. Between 2026-10-23 and 2026-10-28 that step was attempted repeatedly
from a PowerShell prompt and failed every time, never on the ingest itself:

  * A MULTI-LINE PASTE COLLAPSES. Pasting several commands at once put the
    connection string on the end of the `Read-Host` line, and the NEXT command
    was consumed as the answer to the prompt -- so DATABASE_URL was literally
    set to `if ($env:DATABASE_URL) { ... }` and psycopg reported
    `missing "=" after "if"`.
  * THE WRONG URL. `DATABASE_URL` on Railway points at
    `postgres.railway.internal`, which resolves only inside Railway's network.
    From a laptop it can never connect. `DATABASE_PUBLIC_URL` is the one that
    works, via the TCP proxy.
  * `$env:` IS PER-WINDOW. A variable set in one terminal is gone in the next,
    so the ingest ran without it and reported `fe_sendauth: no password
    supplied`.
  * PLACEHOLDERS PASTED LITERALLY. `PASTE_HERE`, `...` and
    `C:\real\path\to\...` were all submitted as real values.

A script has none of those failure modes: one invocation, no pasting into a
prompt mid-sequence, no placeholders, and the variable lives as long as the
script does. It also verifies the write instead of trusting the exit code --
on 2026-10-25 the ingest reported "2,644 rows" while the database stayed empty,
and three days of slates then failed with `-> 0 rows`.

USAGE, from the repository root:

    powershell -ExecutionPolicy Bypass -File scripts\ingest_from_windows.ps1

It will ask for the connection string once, and find the workbook itself if it
is anywhere under your user folder.

    -Xlsx "C:\path\to\export.xlsx"   skip the search
    -DryRun                           parse and report, write nothing
    -Sheet "Team Stats"               if your export's tab is named differently
    -Repo "C:\path\to\bucket"        if run from a copy outside the checkout

It can be run from inside the checkout, from any directory, or from a copy
saved elsewhere -- it looks for the repository in all three places.

RESEARCH_ONLY. This moves team-game and market-line rows. It prices nothing,
places nothing and grades nothing.
#>

[CmdletBinding()]
param(
    [string] $Xlsx = "",
    [switch] $DryRun,
    [string] $Sheet = "",
    # The PropIQ checkout. Normally found automatically; pass it when the
    # script is being run from a copy outside the repository.
    [string] $Repo = "",
    # The public endpoint of the Railway Postgres TCP proxy. Only change this
    # if the proxy is recreated, which gives it a new host and port.
    [string] $ProxyEndpoint = "autorack.proxy.rlwy.net:27751"
)

$ErrorActionPreference = "Stop"

function Say  ([string] $m) { Write-Host $m }
function Good ([string] $m) { Write-Host "  OK   $m" -ForegroundColor Green }
function Bad  ([string] $m) { Write-Host "  FAIL $m" -ForegroundColor Red }
function Note ([string] $m) { Write-Host "       $m" -ForegroundColor DarkGray }

function Invoke-Python {
    <#
      Run python with stdin script text and return its combined output as a
      STRING, never as an error.

      WHY THIS WRAPPER. With $ErrorActionPreference = "Stop", redirecting a
      native command's stderr with `2>&1` is raised as a terminating error in
      Windows PowerShell 5.1. The post-ingest row count is the single most
      important step here, so an aborted script at that point would lose the
      one number this exists to report.
    #>
    param([string] $Script)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = ($Script | & python - 2>&1 | Out-String)
    } finally {
        $ErrorActionPreference = $previous
    }
    return $out.Trim()
}

Say ""
Say "PropIQ workbook ingest - RESEARCH_ONLY"
Say "======================================"
Say ""

# --- [1] the repository ----------------------------------------------------
# THREE PLACES, IN ORDER, because every one of them has actually happened.
# Resolving only from the script's own location broke when the script was run
# from a Downloads copy rather than from inside the checkout, and resolving
# only from the current directory broke when the window was sitting at
# C:\Users\<name>.
function Test-PropIQRepo ([string] $candidate) {
    if ([string]::IsNullOrWhiteSpace($candidate)) { return $false }
    return (Test-Path (Join-Path $candidate "scripts\ingest_bigdataball.py"))
}

$candidates = @()
if ($Repo -ne "") { $candidates += $Repo }
# Beside the script: <repo>\scripts\this.ps1
$candidates += (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
# Wherever the window is sitting, and its parent
$candidates += (Get-Location).Path
$candidates += (Split-Path -Parent (Get-Location).Path)
# The conventional checkout
$candidates += (Join-Path $HOME "bucket")

$repo = $null
foreach ($c in $candidates) {
    if (Test-PropIQRepo $c) { $repo = $c; break }
}
if ($null -eq $repo) {
    Bad "could not find the PropIQ checkout"
    Note "looked for scripts\ingest_bigdataball.py in:"
    foreach ($c in ($candidates | Select-Object -Unique)) { Note "  $c" }
    Note ""
    Note "Pass it explicitly:  -Repo `"C:\path\to\bucket`""
    exit 1
}
Set-Location $repo
Good "repository: $repo"

# --- [2] python and psycopg ------------------------------------------------
try {
    $pyv = (& python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>&1 | Out-String).Trim()
} catch {
    Bad "python is not on PATH"
    exit 1
}
Good "python $pyv"

# Only the real run connects, so a dry run does not need the driver and must
# not go to the network to fetch it.
$hasPsycopg = $false
if (-not $DryRun) {
    try {
        & python -c "import psycopg" 2>$null
        if ($LASTEXITCODE -eq 0) { $hasPsycopg = $true }
    } catch { }
    if (-not $hasPsycopg) {
        Note "psycopg is missing; installing it"
        & python -m pip install --quiet "psycopg[binary]"
        & python -c "import psycopg" 2>$null
        if ($LASTEXITCODE -ne 0) { Bad "psycopg still will not import"; exit 1 }
    }
    Good "psycopg importable"
}

# --- [3] the workbook ------------------------------------------------------
if ($Xlsx -eq "") {
    Note "looking for the BigDataBall export under $HOME ..."
    $found = Get-ChildItem -Path $HOME -Recurse -File -Filter "*NBA_Box_Score_Team-Stats*.xlsx" `
                 -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending |
             Select-Object -First 1
    if ($null -eq $found) {
        Bad "no *NBA_Box_Score_Team-Stats*.xlsx found under $HOME"
        Note "pass it explicitly:  -Xlsx `"C:\path\to\export.xlsx`""
        exit 1
    }
    $Xlsx = $found.FullName
}
if (-not (Test-Path $Xlsx)) { Bad "no workbook at $Xlsx"; exit 1 }
$size = [math]::Round((Get-Item $Xlsx).Length / 1MB, 2)
Good "workbook: $Xlsx  ($size MB)"

# --- [3b] a dry run stops here: it needs no database ----------------------
# `scripts/ingest_bigdataball` states it ("A dry run needs no database and
# would have got this far"), so asking for a connection string and probing
# Railway made --dry-run unusable for the one case it is for: checking the
# parse when the database is unreachable.
if ($DryRun) {
    Say ""
    Say "DRY RUN - no database is needed, so none is asked for."
    Say ""
    $dryArgs = @("-m", "scripts.ingest_bigdataball", "--xlsx", $Xlsx, "--dry-run")
    if ($Sheet -ne "") { $dryArgs += @("--sheet", $Sheet) }
    & python @dryArgs
    $dryExit = $LASTEXITCODE
    Say ""
    Say "======================================"
    if ($dryExit -eq 0) {
        Good "parsed cleanly - nothing was written"
        Say "Re-run without -DryRun to persist."
    } else {
        Bad "the parse failed (exit $dryExit); the tail above says why"
    }
    exit $dryExit
}


# --- [4] the connection string --------------------------------------------
# Asked for HERE, inside the script, so there is no sequence of pasted
# commands for a multi-line paste to collapse.
Say ""
Say "Paste DATABASE_PUBLIC_URL from Railway (Postgres service -> Variables)."
Say "It is NOT the one ending in postgres.railway.internal - but if you paste"
Say "that one by mistake, this script rewrites the host for you."
Say ""
$raw = (Read-Host "connection string").Trim()

if ($raw -eq "") { Bad "nothing was entered"; exit 1 }
if ($raw -notmatch '^postgres(ql)?://') {
    Bad "that does not look like a connection string (it must start postgresql://)"
    # DELIBERATELY ECHOES NOTHING. An earlier version printed the first twelve
    # characters to help, which prints a password outright when the pasted
    # value was the password alone -- breaking this script's one promise.
    Note "nothing was echoed, in case what was pasted was a credential."
    Note "Copy DATABASE_PUBLIC_URL from Railway -> Postgres -> Variables."
    exit 1
}

# The internal host can never resolve off-platform; rewrite it to the proxy.
$conn = $raw -replace 'postgres\.railway\.internal:5432', $ProxyEndpoint
if ($conn -ne $raw) { Note "rewrote the internal host to $ProxyEndpoint" }

# Echo the host ONLY. The password is never printed, logged or written to disk.
if ($conn -match '@([^/?]+)') { Good "target host: $($matches[1])" }
else { Bad "could not read a host out of that string"; exit 1 }

$env:DATABASE_URL = $conn

# --- [5] prove the connection before doing any work -----------------------
Say ""
Say "[1/4] connecting ..."
$probe = @'
import os, sys
import psycopg
try:
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=20) as c:
        with c.cursor() as cur:
            cur.execute("select current_database(), current_user, version()")
            db, who, ver = cur.fetchone()
    print(f"OK\t{db}\t{who}\t{ver.split(',')[0]}")
except Exception as exc:
    print(f"ERR\t{type(exc).__name__}\t{exc}")
    sys.exit(1)
'@
$probeOut = Invoke-Python $probe
if ($probeOut -notmatch '^OK') {
    Bad "could not connect"
    Note $probeOut
    Note "A timeout usually means the TCP proxy is gone - check Railway ->"
    Note "Postgres -> Settings -> Networking. An authentication failure means"
    Note "the password was rotated; copy DATABASE_PUBLIC_URL again."
    exit 1
}
$parts = $probeOut -split "`t"
Good "connected to database '$($parts[1])' as '$($parts[2])' ($($parts[3]))"
$targetDb = $parts[1]

# --- [6] the row count BEFORE ---------------------------------------------
$countScript = @'
import os, sys
import psycopg
with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=20) as c:
    with c.cursor() as cur:
        out = []
        for t in ("team_game_stats", "game_market_lines", "player_game_logs"):
            try:
                cur.execute(f"select count(*) from {t}")
                out.append(f"{t}={cur.fetchone()[0]}")
            except Exception as exc:
                c.rollback()
                out.append(f"{t}=<{type(exc).__name__}>")
        print("\t".join(out))
'@
Say ""
Say "[2/4] counting rows before ..."
$before = Invoke-Python $countScript
Note $before

# --- [7] the ingest -------------------------------------------------------
# A dry run has already exited above, so this is always the real thing.
$ingestArgs = @("-m", "scripts.ingest_bigdataball", "--xlsx", $Xlsx)
if ($Sheet -ne "") { $ingestArgs += @("--sheet", $Sheet) }

Say ""
Say "[3/4] ingesting ..."
Say ""

# Streamed so the whole tail is visible. The exit code is kept, but it is
# NOT what this script trusts - step 4 is.
& python @ingestArgs
$ingestExit = $LASTEXITCODE
Say ""
if ($ingestExit -eq 0) { Good "ingest exited 0" }
else                   { Bad  "ingest exited $ingestExit" }

# --- [8] VERIFY the write, which is the whole point ----------------------
Say ""
Say "[4/4] counting rows after ..."
$after = Invoke-Python $countScript
Note $after

Say ""
Say "======================================"

# THE DELTA DECIDES, not the exit code and not the absolute count.
#
# `-gt 0` was wrong twice over: it reported success on an already-populated
# table, so a rerun that wrote nothing -- or failed outright -- still exited 0
# as long as some earlier run had left rows behind. The exit code is no better:
# on 2026-10-25 an ingest reported "2,644 rows / 1,322 games" and exited 0
# while the table stayed empty. So both counts are parsed and compared.
function Get-Count ([string] $text, [string] $table) {
    if ($text -match "$table=(\d+)") { return [int]$matches[1] }
    return -1
}

$teamBefore = Get-Count $before "team_game_stats"
$teamAfter  = Get-Count $after  "team_game_stats"

if ($teamBefore -lt 0 -or $teamAfter -lt 0) {
    Bad "could not read a team_game_stats count, so the write is unverified"
    Note "before: $before"
    Note "after : $after"
    exit 1
}

$delta = $teamAfter - $teamBefore

if ($delta -gt 0) {
    Good "team_game_stats went from $teamBefore to $teamAfter (+$delta) in '$targetDb'"
    Say ""
    Say "The 09:00 PT slate reads these tables. The next run should build a"
    Say "board instead of exiting 1 - though it will still abstain on every"
    Say "row until a model artifact exists, which is the separate blocker."
    exit 0
}

if ($delta -eq 0 -and $teamAfter -gt 0) {
    # Not necessarily a failure: the upsert is idempotent, so re-ingesting the
    # same workbook legitimately changes nothing. Said plainly either way,
    # because "0 new rows" and "it worked" are different sentences.
    Write-Host "  NOTE team_game_stats holds $teamAfter row(s) and did not change." -ForegroundColor Yellow
    Say ""
    Say "The upsert is idempotent, so re-ingesting the SAME workbook correctly"
    Say "adds nothing. That is fine if these rows are the ones you wanted."
    Say "If you expected new rows, the workbook holds games already ingested -"
    Say "check the date range in the tail above."
    if ($ingestExit -ne 0) {
        Bad "but the ingest exited $ingestExit, so something did fail"
        exit 1
    }
    exit 0
}

Bad "team_game_stats is still empty after the ingest (before=$teamBefore after=$teamAfter)"
Say ""
Say "This is the exact failure that wasted 2026-10-23 to 2026-10-28: an ingest"
Say "that reports success while the table stays empty. The tail above is the"
Say "evidence - look for an 'Upserted N team-game stat rows' line. If it is"
Say "absent, the write never reached a commit. If it is present, this ran"
Say "against a different database than the worker reads ('$targetDb' here)."
exit 1
