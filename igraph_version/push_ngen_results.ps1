# push_ngen_results.ps1
# Commits and pushes ngen sweep CSVs to git every hour until all 24 are present.

$REPO     = "C:\Users\lucsc\Thesis\grad\grad\igraph_version"
$RESULTS  = "$REPO\results"
$TOTAL    = 24   # 6 n_gen values x 4 methods
$INTERVAL = 3600 # seconds between pushes

$ngen_values = @(100, 200, 400, 800, 1600, 3200)
$methods     = @("diffusion", "gan", "graphsmote", "diga")

function Push-Results {
    $csvs = @()
    foreach ($n in $ngen_values) {
        foreach ($m in $methods) {
            $f = "$RESULTS\classifier_comparison_ethereum_ngen_${n}_${m}.csv"
            if (Test-Path $f) { $csvs += $f }
        }
    }

    if ($csvs.Count -eq 0) {
        Write-Host "$(Get-Date -Format 'HH:mm:ss')  No new CSVs yet, skipping commit."
        return $false
    }

    git -C $REPO add ($csvs | ForEach-Object { $_.Replace("$REPO\", "") })
    $status = git -C $REPO status --porcelain
    if (-not $status) {
        Write-Host "$(Get-Date -Format 'HH:mm:ss')  Nothing new to commit."
        return $csvs.Count -ge $TOTAL
    }

    $msg = "experiment: ngen sweep results ($($csvs.Count)/$TOTAL CSVs)"
    git -C $REPO commit -m $msg
    git -C $REPO push origin main
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  Pushed $($csvs.Count)/$TOTAL CSVs."
    return $csvs.Count -ge $TOTAL
}

Write-Host "Starting ngen sweep auto-push (every $($INTERVAL/60) min, target $TOTAL CSVs) ..."

while ($true) {
    $done = Push-Results
    if ($done) {
        Write-Host "$(Get-Date -Format 'HH:mm:ss')  All $TOTAL CSVs pushed. Done."
        break
    }
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  Sleeping $($INTERVAL/60) minutes ..."
    Start-Sleep -Seconds $INTERVAL
}
