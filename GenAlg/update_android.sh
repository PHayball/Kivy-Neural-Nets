#!/bin/bash

cp -u ./kivy-neuralnet.py ./android/main.py
cp -u ./neuralnet.kv ./android
cp -u ./buildozer.spec ./android
cd android
buildozer android debug deploy run


