param(
    [ValidateSet('preparar', 'demo', 'salvas', 'verificar', 'baixar', 'mcp', 'preparar-av', 'verificar-av', 'extrair-av', 'app', 'prever-video')]
    [string]$Acao = 'demo',
    # Para 'extrair-av' e 'prever-video': caminho do arquivo de vídeo local a processar (offline, sem Gemini nem YouTube).
    [string]$Video = '',
    # Só para 'prever-video': channel_id de um canal da base curada e setor (beleza ou automotivo).
    [string]$Canal = '',
    [string]$Setor = '',
    [string]$Historico = '',
    [ValidateSet('', 'marca', 'criador')]
    [string]$TipoCanal = ''
)

# Windows nativo. O ambiente Linux .venv nunca e reutilizado ou substituido.
$ErrorActionPreference = 'Stop'
$raizGalaxy = $PSScriptRoot
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $raizGalaxy '.windows\cache-uv' }
if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $raizGalaxy '.windows\python' }
$ferramentasGalaxy = Join-Path $raizGalaxy '.windows\ferramentas'
$uvGalaxy = Join-Path $ferramentasGalaxy 'uv.exe'
if (-not (Test-Path -LiteralPath $uvGalaxy)) {
    $comandoUvGalaxy = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($comandoUvGalaxy) { $uvGalaxy = $comandoUvGalaxy.Source }
}
if (-not (Test-Path -LiteralPath $uvGalaxy)) {
    $uvUsuarioGalaxy = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path -LiteralPath $uvUsuarioGalaxy) { $uvGalaxy = $uvUsuarioGalaxy }
}
if (-not (Test-Path -LiteralPath $uvGalaxy)) {
    if ($Acao -ne 'preparar') {
        throw 'uv para Windows ausente. Execute primeiro: powershell -NoProfile -ExecutionPolicy Bypass -File .\executar_windows.ps1 preparar'
    }
    New-Item -ItemType Directory -Force -Path $ferramentasGalaxy | Out-Null
    $instaladorGalaxy = Join-Path $ferramentasGalaxy 'instalar_uv.ps1'
    Write-Host 'Baixando o instalador oficial do uv para Windows (sem alterar o PATH global).'
    Invoke-WebRequest -Uri 'https://astral.sh/uv/0.12.23/install.ps1' -OutFile $instaladorGalaxy -UseBasicParsing
    $env:UV_UNMANAGED_INSTALL = $ferramentasGalaxy
    & $instaladorGalaxy
    if (-not (Test-Path -LiteralPath $uvGalaxy)) { throw 'A instalacao do uv nao produziu uv.exe.' }
}

if (-not $env:UV_PROJECT_ENVIRONMENT) {
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $raizGalaxy '.venv-windows'
}
if ([IO.Path]::GetFullPath($env:UV_PROJECT_ENVIRONMENT).TrimEnd('\') -eq (Join-Path $raizGalaxy '.venv')) {
    throw 'Use .venv-windows para Windows; .venv esta reservado ao ambiente WSL.'
}
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Push-Location $raizGalaxy
try {
    switch ($Acao) {
        'preparar' { & $uvGalaxy sync --locked }
        'demo' { & $uvGalaxy run --locked casegalaxy-demo }
        'salvas' { & $uvGalaxy run --locked casegalaxy-demo --mostrar-salvas }
        'verificar' { & $uvGalaxy run --locked casegalaxy-demo --verificar-ambiente }
        'baixar' { & $uvGalaxy run --locked casegalaxy-demo --baixar-codificadores }
        'mcp' { & $uvGalaxy run --locked casegalaxy-mcp }
        # Camada audiovisual local: preparar-av baixa os pesos (único passo com rede); verificar-av confere SHA-256 sem rede.
        'preparar-av' { & $uvGalaxy run --locked casegalaxy-preparar-av }
        'verificar-av' { & $uvGalaxy run --locked casegalaxy-preparar-av --verificar }
        # Interface web local (somente 127.0.0.1): abre o navegador padrão; nenhuma análise roda antes do clique em Analisar.
        'app' { & $uvGalaxy run --locked casegalaxy-interface --abrir-navegador }
        # Previsão de um vídeo local pela rota local, offline: exige o vídeo, um canal da base e o setor.
        'prever-video' {
            if (-not $Video -or -not $Canal -or -not $Setor) { throw 'Informe: ... executar_windows.ps1 prever-video -Video C:\caminho\video.mp4 -Canal UCxxxxxxxxxxxxxxxxxxxxxx -Setor beleza' }
            $argsPrevisaoGalaxy = @("--video", $Video, "--canal", $Canal, "--setor", $Setor)
            if ($Historico) { $argsPrevisaoGalaxy += @("--historico", $Historico) }
            if ($TipoCanal) { $argsPrevisaoGalaxy += @("--tipo-canal", $TipoCanal) }
            & $uvGalaxy run --locked casegalaxy-demo-local @argsPrevisaoGalaxy
        }
        'extrair-av' {
            if (-not $Video) { throw 'Informe o video: ... executar_windows.ps1 extrair-av -Video C:\caminho\video.mp4' }
            & $uvGalaxy run --locked casegalaxy-extrair-av $Video
        }
    }
    $codigoGalaxy = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $codigoGalaxy
