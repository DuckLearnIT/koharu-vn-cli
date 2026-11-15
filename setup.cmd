@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

cd /d "%~dp0"

echo [1/4] Kiem tra uv...
where uv >nul 2>&1
if errorlevel 1 (
    echo Dang cai dat uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [LOI] Khong cai duoc uv!
        exit /b 1
    )
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)

echo [2/4] Cai dat Python 3.12 va dong bo moi truong...
uv python install 3.12
if errorlevel 1 exit /b 1

uv sync --locked
if errorlevel 1 exit /b 1

echo [3/4] Cai dat lenh koharu-vn toan cuc...
uv tool update-shell >nul 2>&1
uv tool install --force .
if errorlevel 1 exit /b 1

set "KOHARU_EXE=%USERPROFILE%\.local\bin\koharu-vn.exe"
if not exist "%KOHARU_EXE%" set "KOHARU_EXE=koharu-vn"

echo [4/4] Tai truoc cac model ONNX (RT-DETR + CTD + LaMa-Manga)...
"%KOHARU_EXE%" models --accurate-detector
if errorlevel 1 (
    echo [CANH BAO] Chua tai duoc model ONNX luc cai dat, app se tu tai khi chay lan dau.
)

echo.
echo ========================================================
echo   CAI DAT HOAN TAT!
echo   Hay mo cua so cmd moi va go: koharu-vn
echo ========================================================
endlocal
