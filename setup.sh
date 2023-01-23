#!/bin/bash
# this script sets the tenet application for it to work properly

ROOT_DIR=~/.tenet/

# the two main directories are: config and plugins
mkdir -p "${ROOT_DIR}/config"
mkdir -p "${ROOT_DIR}/plugins"

# the configuration file and plugin scripts must be correctly placed in their respective location
cp -a tenet/config/. "${ROOT_DIR}/config"
cp -a tenet/plugins/* "${ROOT_DIR}/plugins"
