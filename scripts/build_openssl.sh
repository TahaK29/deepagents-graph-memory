#!/usr/bin/env bash
# Build-only dependency for manylinux_2_28; auditwheel bundles it into the wheel.
set -euo pipefail
dnf install -y perl-IPC-Cmd perl-Pod-Html
curl -fsSL https://github.com/openssl/openssl/releases/download/openssl-3.5.8/openssl-3.5.8.tar.gz -o /tmp/graph-memory-openssl.tar.gz
echo 'a8f84a39918ec6415ce765d9b429d313ba97b8143169c172e734b9514464f5b2  /tmp/graph-memory-openssl.tar.gz' | sha256sum -c -
mkdir -p /tmp/graph-memory-openssl-src
tar -xzf /tmp/graph-memory-openssl.tar.gz -C /tmp/graph-memory-openssl-src --strip-components=1
cd /tmp/graph-memory-openssl-src
./Configure --prefix=/opt/graph-memory-openssl --libdir=lib shared no-tests
make -j2 > /tmp/graph-memory-openssl-build.log
make install_sw > /tmp/graph-memory-openssl-install.log
