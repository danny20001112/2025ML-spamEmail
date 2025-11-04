@echo off
echo =================================
echo 正在啟動垃圾郵件分類系統...
echo =================================

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 未安裝 Python，請先安裝 Python
    pause
    exit /b 1
)

REM 檢查 streamlit 是否安裝
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安裝 streamlit...
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    if errorlevel 1 (
        echo [錯誤] Streamlit 安裝失敗
        pause
        exit /b 1
    )
)

echo [信息] 正在啟動應用程式...
python -m streamlit run app.py

if errorlevel 1 (
    echo [錯誤] 應用程式啟動失敗
    pause
    exit /b 1
)

pause
