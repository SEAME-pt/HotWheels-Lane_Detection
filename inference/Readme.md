## Compile for jetson :
mkdir build_jetson && cd build_jetson
make sure it's empty (maybe rm -rf inside)
~/qtjetson/qt5.5.12/bin/qmake .../trt_inference.pro
make
scp trt_inference $JETSON:/home/hotweels/dev/model_loader

## Then in jetson :
cd /home/hotweels/dev/model_loader
./trt_inference