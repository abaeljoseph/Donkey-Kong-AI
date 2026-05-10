@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if %ERRORLEVEL% neq 0 (
    echo vcvarsall failed
    exit /b 1
)
echo MSVC environment activated
cl.exe 2>&1 | findstr /i "version"
python -m pip install stable-retro
echo DONE exitcode=%ERRORLEVEL%
