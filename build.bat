@echo off
setlocal
if not exist build mkdir build
cd build
@REM cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_OPENGL=OFF
cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_OPENGL=ON
cmake --build . --config Debug
endlocal