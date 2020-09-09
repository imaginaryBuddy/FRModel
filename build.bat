activate base
powershell "conda-build . | tee build.log"
pause