#!/bin/bash

mkdir -p ./android

cp -u ./kivy-neuralnet.py ./android/main.py
cp -u ./neuralnet.kv ./android
cd android/
buildozer init
# mkdir -p ./.buildozer/android/platform
buildozer android debug deploy run
cp -ru ~/Scripts/Kivy/python-for-android ./.buildozer/android/platform/
cp ../buildozer.spec ./
buildozer android debug deploy run


