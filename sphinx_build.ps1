conda activate FRModel
cd .\src\sphinx\
sphinx-build -b html source build
cd ..\..\
xcopy /s .\src\sphinx\build .\docs
a
pause