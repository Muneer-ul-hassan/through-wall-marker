#!/bin/sh
# Make a self-signed TLS certificate for the wall-marker server so an
# iPhone on the LAN can open https://<laptop-LAN-IP>:<port>/viewer
# (Safari needs a secure context for getUserMedia on a LAN IP; plain
# http works on localhost only).
#
# Usage:
#   tools/make_cert.sh [LAN-IP-or-hostname] [out-dir]
# Defaults: hostname of this machine, ./certs
# Then start the server with:
#   python -m server.main --mode virtual --scenario s1_basic \
#       --https --cert certs/cert.pem --key certs/key.pem
# On the iPhone, open the https URL and accept the certificate warning
# (self-signed: Safari will show "Not Secure" / ask to trust it).
set -eu
HOST="${1:-$(hostname)}"
OUT="${2:-certs}"
mkdir -p "$OUT"
openssl req -x509 -newkey rsa:2048 -keyout "$OUT/key.pem" \
    -out "$OUT/cert.pem" -days 365 -nodes \
    -subj "/CN=$HOST" \
    -addext "subjectAltName=DNS:$HOST,DNS:localhost,IP:127.0.0.1"
echo "Wrote $OUT/cert.pem and $OUT/key.pem (CN=$HOST, valid 365 days)"
