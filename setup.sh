#!/bin/bash
# this script sets the securityaware application for it to work properly

ROOT_DIR=~/.securityaware/

# the two main directories are: config and plugins
mkdir -p "${ROOT_DIR}/config"
mkdir -p "${ROOT_DIR}/plugins/nexus"

# the configuration file and plugin scripts must be correctly placed in their respective location
cp config/securityaware.yml "${ROOT_DIR}/config"
cp securityaware/plugins/* "${ROOT_DIR}/plugins"