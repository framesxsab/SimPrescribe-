[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SourceDatabaseUrl,

    [Parameter(Mandatory)]
    [string]$RestoreDatabaseUrl,

    [Parameter(Mandatory)]
    [string]$BackupPath
)

$ErrorActionPreference = "Stop"

function Convert-PostgresUrl([string]$DatabaseUrl) {
    return $DatabaseUrl -replace '^postgresql\+psycopg2?:', 'postgresql:'
}

function Get-PostgresTarget([string]$DatabaseUrl) {
    $uri = [System.Uri]$DatabaseUrl
    $databaseName = [System.Uri]::UnescapeDataString($uri.AbsolutePath.TrimStart('/'))
    if (-not $uri.Host -or -not $databaseName) {
        throw "Database URL must include a host and database name."
    }
    $port = if ($uri.IsDefaultPort) { 5432 } else { $uri.Port }
    return @{
        Identity = "$($uri.Host.ToLowerInvariant()):$port/$databaseName"
        DatabaseName = $databaseName
    }
}

$sourceUrl = Convert-PostgresUrl $SourceDatabaseUrl
$restoreUrl = Convert-PostgresUrl $RestoreDatabaseUrl
$sourceTarget = Get-PostgresTarget $sourceUrl
$restoreTarget = Get-PostgresTarget $restoreUrl

if ($sourceTarget.Identity -eq $restoreTarget.Identity) {
    throw "Restore target must differ from source database."
}
if ($restoreTarget.DatabaseName -notmatch '(?i)(restore|verify|test)') {
    throw "Restore target name must include restore, verify, or test. Refusing a potentially production target."
}
if (Test-Path -LiteralPath $BackupPath) {
    throw "Backup path already exists. Choose a new immutable backup path instead of overwriting it."
}
if (-not (Test-Path -LiteralPath (Split-Path -Parent $BackupPath))) {
    throw "Backup directory does not exist."
}

foreach ($command in "pg_dump", "pg_restore", "psql") {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "$command is required. Install PostgreSQL client tools before running recovery verification."
    }
}

& pg_dump --format=custom "--file=$BackupPath" $sourceUrl
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed." }

& pg_restore --list $BackupPath | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Backup archive validation failed." }

# ponytail: restore only accepts an explicitly named disposable target; use a managed restore workflow if finer-grained recovery is needed.
& pg_restore --clean --if-exists --no-owner --no-privileges "--dbname=$restoreUrl" $BackupPath
if ($LASTEXITCODE -ne 0) { throw "pg_restore failed." }

$result = & psql $restoreUrl --tuples-only --no-align --command "SELECT CASE WHEN to_regclass('public.analyses') IS NULL THEN 'missing' ELSE 'ready' END;"
if ($LASTEXITCODE -ne 0 -or $result -notmatch 'ready') {
    throw "Restored database did not contain the analyses table."
}

Write-Host "Recovery verification passed. Preserve the backup in encrypted managed storage and record the verification date."
