# Production recovery verification

Run this only from a secure operator workstation with PostgreSQL client tools installed. It creates a new custom-format backup, validates the archive, restores it into a disposable database, and confirms the `analyses` table exists.

The restore URL must name a database containing `restore`, `verify`, or `test`; the script rejects all other targets and never overwrites an existing backup file.

```powershell
.\scripts\verify-postgres-recovery.ps1 `
  -SourceDatabaseUrl $env:DATABASE_URL `
  -RestoreDatabaseUrl $env:RESTORE_DATABASE_URL `
  -BackupPath "D:\encrypted-backups\simpliscribe-$(Get-Date -Format yyyyMMdd).dump"
```

Requirements before running:

1. `RESTORE_DATABASE_URL` points to a disposable private database, never production.
2. `D:\encrypted-backups` is encrypted managed storage with restricted operator access.
3. Source and restore credentials come from the deployment secret manager, never this repository.
4. Record the date, operator, source backup identifier, restore target, and result in the approved operations system; do not put identifiable data in that record.

This proves archive creation and full schema/data restore for the configured database. It does not prove application-level clinical correctness, identity-provider availability, or disaster-region recovery.
