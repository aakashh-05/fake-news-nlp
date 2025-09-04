
@echo off
REM Prepare data (change input if you have your own CSV)
python src\prepare_data.py --input data\raw\sample.csv
IF %ERRORLEVEL% NEQ 0 GOTO :error

REM Train TF-IDF baseline
python src\train_tfidf.py
IF %ERRORLEVEL% NEQ 0 GOTO :error

REM Train GloVe baseline
python src\train_glove.py
IF %ERRORLEVEL% NEQ 0 GOTO :error

REM Train BERT (optional, comment out if you want)
python src\train_bert.py
IF %ERRORLEVEL% NEQ 0 GOTO :error

REM Evaluate all
python src\evaluate.py
IF %ERRORLEVEL% NEQ 0 GOTO :error

echo All done!
GOTO :eof

:error
echo Something failed. Check the console above for details.
exit /b 1
