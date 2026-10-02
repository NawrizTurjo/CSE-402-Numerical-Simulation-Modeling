# Build 2105032.pdf, package 2105032_LaTeX_Source.zip, and verify that the zip
# compiles from scratch in a clean temporary directory.
# Usage (from the Assignment folder):  pwsh ./build.ps1
$ErrorActionPreference = 'Stop'
$root   = $PSScriptRoot
$src    = Join-Path $root '2105032_LaTeX_Source'
$name   = '2105032'
$zip    = Join-Path $root "$($name)_LaTeX_Source.zip"
$pdfOut = Join-Path $root "$name.pdf"

function Invoke-TexBuild([string]$dir) {
    Push-Location $dir
    try {
        pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "pdflatex (pass 1) failed in $dir" }
        bibtex $name | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "bibtex failed in $dir" }
        foreach ($i in 1..2) {
            pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "pdflatex (pass $($i + 1)) failed in $dir" }
        }
    } finally { Pop-Location }
}

# 1. Clean build in the source folder
Write-Host '== Building deck'
Invoke-TexBuild $src
Copy-Item (Join-Path $src "$name.pdf") $pdfOut -Force

# 2. Stage the source tree (no build by-products except the .bbl)
Write-Host '== Packaging'
$stageRoot = Join-Path ([System.IO.Path]::GetTempPath()) "gd_stage_$([guid]::NewGuid().ToString('N'))"
$stage     = Join-Path $stageRoot "$($name)_LaTeX_Source"
New-Item -ItemType Directory -Force $stage | Out-Null
$exclude = '\.(aux|log|nav|out|snm|toc|vrb|blg|fls|fdb_latexmk|synctex\.gz|pdf)$'
Get-ChildItem $src -Recurse -File | Where-Object {
    ($_.FullName -notmatch '__pycache__') -and
    (($_.Name -notmatch $exclude) -or ($_.DirectoryName -like '*figures*'))
} | ForEach-Object {
    $rel  = $_.FullName.Substring($src.Length + 1)
    $dest = Join-Path $stage $rel
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Copy-Item $_.FullName $dest
}
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $stage -DestinationPath $zip
Remove-Item $stageRoot -Recurse -Force

# 3. Clean-room test: unzip into a fresh temp dir and compile
Write-Host '== Clean-room compile test'
$test = Join-Path ([System.IO.Path]::GetTempPath()) "gd_test_$([guid]::NewGuid().ToString('N'))"
Expand-Archive $zip -DestinationPath $test
$testSrc = Join-Path $test "$($name)_LaTeX_Source"
Invoke-TexBuild $testSrc
$a = (Get-Item $pdfOut).Length
$b = (Get-Item (Join-Path $testSrc "$name.pdf")).Length
$pagesA = (Select-String -Path (Join-Path $src "$name.log") -Pattern 'Output written.*\((\d+) pages').Matches[0].Groups[1].Value
$pagesB = (Select-String -Path (Join-Path $testSrc "$name.log") -Pattern 'Output written.*\((\d+) pages').Matches[0].Groups[1].Value
Write-Host "Submitted PDF: $pagesA pages ($a bytes); clean-room PDF: $pagesB pages ($b bytes)"
if ($pagesA -ne $pagesB) { throw 'Page counts differ between submitted and clean-room builds' }
Remove-Item $test -Recurse -Force
Write-Host "OK -> $pdfOut"
Write-Host "OK -> $zip"
