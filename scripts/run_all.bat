@echo off
setlocal
set "ENV_NAME=MEDICAL"
set "CONFIG=configs/smoke.yaml"
set "RUN_ID=smoke_pipeline_acceptance"
cd /d "%~dp0\.."
conda run -n %ENV_NAME% python -m compileall src tests || exit /b 1
conda run -n %ENV_NAME% ruff check . || exit /b 1
conda run -n %ENV_NAME% pytest -q || exit /b 1
conda run -n %ENV_NAME% python -m virtual_staining.cli run-pipeline --config %CONFIG% --run-id %RUN_ID% || exit /b 1
endlocal

