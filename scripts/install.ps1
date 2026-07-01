# ============================================================
# Miaowa Code — 一键安装脚本 (Windows PowerShell)
# ============================================================
$ErrorActionPreference = "Stop"

Write-Host "🐱 Miaowa Code 安装脚本" -ForegroundColor Green
Write-Host "========================================"

# 检查 Python 版本
$pythonCmd = $null
foreach ($cmd in @("python3.12", "python3.11", "python3.10", "python3", "python")) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found) {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        $major = [int]($ver.Split('.')[0])
        $minor = [int]($ver.Split('.')[1])
        if ($major -ge 3 -and $minor -ge 10) {
            $pythonCmd = $cmd
            break
        }
    }
}

if (-not $pythonCmd) {
    Write-Host "错误: 未找到 Python >= 3.10" -ForegroundColor Red
    Write-Host "请先安装 Python 3.10+: https://www.python.org/downloads/"
    exit 1
}

Write-Host "✓ Python: $(& $pythonCmd --version)" -ForegroundColor Green

# 检查/安装 Poetry
$poetryFound = Get-Command poetry -ErrorAction SilentlyContinue
if (-not $poetryFound) {
    Write-Host "正在安装 Poetry..." -ForegroundColor Yellow
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | & $pythonCmd -
}

Write-Host "✓ Poetry: $(poetry --version)" -ForegroundColor Green

# 安装依赖
Write-Host "正在安装项目依赖..."
poetry install --with dev

# 配置环境变量
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "⚠ 已创建 .env 文件，请编辑填入 DeepSeek API Key:" -ForegroundColor Yellow
    Write-Host "   notepad .env"
}

Write-Host ""
Write-Host "✅ 安装完成！" -ForegroundColor Green
Write-Host "运行 'poetry run miaowa' 启动 Miaowa Code"
