cd build
#cmake -DCMAKE_PREFIX_PATH=../../libtorch ..
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build . --config Release
