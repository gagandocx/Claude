@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: CCT Rectangle Bot - One-Click Updater
:: ============================================================
:: Downloads the latest files from GitHub and replaces all
:: files in the local folder with fresh copies.
:: ============================================================

echo.
echo ============================================================
echo    CCT Rectangle Bot - Updater
echo ============================================================
echo.

:: Configuration
set "LOCAL_PATH=F:\Automation\EA Testing\CCT_Rectangle_Bot\v1"
set "REPO_URL=https://github.com/gagandocx/Claude.git"
set "BRANCH=feat/cct-rectangle-bot-v2"
set "TEMP_DIR=%TEMP%\cct_update_temp_%RANDOM%"
set "SUBFOLDER=CCT_Rectangle_Bot"

:: Check if git is available
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/downloads
    goto :error
)

:: Check if pip is available
where pip >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] pip is not found in PATH. Will skip requirements install.
    set "SKIP_PIP=1"
) else (
    set "SKIP_PIP=0"
)

:: Step 1: Clone the repository to a temp folder
echo [1/5] Cloning latest files from GitHub...
echo       Branch: %BRANCH%
echo.

if exist "%TEMP_DIR%" (
    rmdir /s /q "%TEMP_DIR%"
)

git clone --branch %BRANCH% --single-branch --depth 1 "%REPO_URL%" "%TEMP_DIR%"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Git clone failed. Check your internet connection and try again.
    goto :error
)

echo.
echo [2/5] Clone successful. Preparing to update local files...

:: Verify the subfolder exists in the cloned repo
if not exist "%TEMP_DIR%\%SUBFOLDER%" (
    echo.
    echo [ERROR] Subfolder "%SUBFOLDER%" not found in the repository.
    goto :cleanup_error
)

:: Step 2: Create the local directory if it doesn't exist
if not exist "%LOCAL_PATH%" (
    echo       Creating local directory...
    mkdir "%LOCAL_PATH%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create directory: %LOCAL_PATH%
        goto :cleanup_error
    )
)

:: Step 3: Remove old files from local folder (but keep the folder itself)
echo [3/5] Removing old files from local folder...
echo       Path: %LOCAL_PATH%

:: Delete all files in the target directory
del /q "%LOCAL_PATH%\*" 2>nul

:: Delete all subdirectories in the target directory
for /d %%D in ("%LOCAL_PATH%\*") do (
    rmdir /s /q "%%D" 2>nul
)

:: Step 4: Copy new files from temp to local folder
echo [4/5] Copying fresh files to local folder...

xcopy "%TEMP_DIR%\%SUBFOLDER%\*" "%LOCAL_PATH%\" /e /i /y /q
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to copy files to local folder.
    goto :cleanup_error
)

:: Verify live_mt5 subfolder was copied
if exist "%LOCAL_PATH%\live_mt5" (
    echo       live_mt5 subfolder updated successfully.
) else (
    echo [WARNING] live_mt5 subfolder was not found in the update.
)

:: Step 5: Install/update pip requirements
echo [5/5] Installing/updating pip requirements...

if "%SKIP_PIP%"=="1" (
    echo       [SKIPPED] pip not found. Install Python and pip, then run:
    echo       pip install -r "%LOCAL_PATH%\requirements.txt"
) else (
    if exist "%LOCAL_PATH%\requirements.txt" (
        pip install -r "%LOCAL_PATH%\requirements.txt" --upgrade
        if %ERRORLEVEL% neq 0 (
            echo [WARNING] Some pip packages may have failed to install.
            echo          You can manually run: pip install -r "%LOCAL_PATH%\requirements.txt"
        ) else (
            echo       All pip requirements installed successfully.
        )
    ) else (
        echo       No requirements.txt found. Skipping pip install.
    )
)

:: Cleanup temp folder
echo.
echo Cleaning up temporary files...
rmdir /s /q "%TEMP_DIR%" 2>nul

:: Success
echo.
echo ============================================================
echo    UPDATE COMPLETE
echo ============================================================
echo.
echo All files have been updated in:
echo %LOCAL_PATH%
echo.
echo ============================================================
goto :done

:cleanup_error
echo.
echo Cleaning up temporary files...
rmdir /s /q "%TEMP_DIR%" 2>nul

:error
echo.
echo ============================================================
echo    UPDATE FAILED - See error messages above
echo ============================================================
echo.

:done
echo.
pause
