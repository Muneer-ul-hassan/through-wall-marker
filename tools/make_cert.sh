#!/bin/sh
# Make TLS certificates for the wall-marker server so an iPhone on the
# LAN can open https://<laptop-LAN-IP>:8443/viewer and actually trust
# it (Safari needs a secure context for getUserMedia on a LAN IP;
# plain http works on localhost only).
#
# A bare self-signed leaf does NOT work on iOS, for two independent
# reasons (Apple primary sources):
#   (1) Since iOS 13 Safari ignores the certificate CN and requires
#       the exact host/IP typed in the URL in the subjectAltName
#       (SAN) — so the leaf must carry IP:192.168.10.2 (the laptop's
#       Tello-network IP), not just a CN.
#   (2) iOS offers no trust switch for a self-signed leaf at all.
#       Only a ROOT CA installed on the phone can be fully trusted
#       (Settings > General > About > Certificate Trust Settings).
# So this script creates a LOCAL CA and issues a short-lived leaf
# signed by it. Install the CA public cert (ca.crt) on the phone
# once; the leaf is then trusted with no warning.
#
# Usage:
#   tools/make_cert.sh [LAN-IP-or-hostname] [out-dir]
# Defaults: 192.168.10.2 (the laptop's Tello-network IP), ./certs
#   - If the first arg is an IPv4 address it becomes the LAN IP in
#     the leaf SAN (the hostname is taken from `hostname`).
#   - If it is a hostname it becomes the DNS name in the SAN and the
#     LAN IP stays 192.168.10.2.
# The leaf SAN always also covers IP:127.0.0.1, DNS:localhost, and
# IP:192.168.137.1 (the Windows Mobile Hotspot IP, so the same cert
# covers the hotspot fallback).
# Re-running the script REUSES an existing CA in the out-dir (only
# the leaf is re-issued), so the CA already trusted on the phone
# stays valid. Delete ca.key/ca.crt to start a fresh CA — then the
# phone must install the new ca.crt.
#
# Then start the server with:
#   python -m server.main --mode live --port 8443 \
#       --https --cert certs/cert.pem --key certs/key.pem
# On the iPhone: AirDrop or email certs/ca.crt to the phone (the CA
# PUBLIC cert only — NEVER send ca.key anywhere), tap it,
# Settings > Profile Downloaded > Install, then
# Settings > General > About > Certificate Trust Settings >
# switch the "Wall-Marker Local CA" on. Then open
# https://192.168.10.2:8443/viewer — no warning.
set -eu

ARG1="${1:-192.168.10.2}"
OUT="${2:-certs}"

# IPv4 address (digits and dots only) vs hostname.
case "$ARG1" in
    *[!0-9.]*)
        HOSTNAME="$ARG1"
        LAN_IP="192.168.10.2"
        ;;
    *)
        LAN_IP="$ARG1"
        HOSTNAME="$(hostname)"
        ;;
esac

# SAN: the LAN IP, the Tello-network IP, loopback, hotspot IP,
# localhost and the hostname (de-duplicated).
SAN="IP:127.0.0.1,IP:192.168.137.1,DNS:localhost,DNS:$HOSTNAME"
case "$LAN_IP" in
    192.168.10.2)
        SAN="IP:192.168.10.2,$SAN"
        ;;
    *)
        SAN="IP:$LAN_IP,IP:192.168.10.2,$SAN"
        ;;
esac

mkdir -p "$OUT"

# --- Local CA (created once, reused on later runs) -------------------
if [ -f "$OUT/ca.key" ] && [ -f "$OUT/ca.crt" ]; then
    echo "Reusing existing CA in $OUT (phone trust stays valid)"
else
    echo "Creating local CA (RSA 2048, SHA-256, 10 years)..."
    openssl req -x509 -newkey rsa:2048 -keyout "$OUT/ca.key" \
        -out "$OUT/ca.crt" -days 3650 -nodes -sha256 \
        -subj "/CN=Wall-Marker Local CA" \
        -addext "basicConstraints=critical,CA:TRUE" \
        -addext "keyUsage=critical,keyCertSign,cRLSign"
    echo "Wrote $OUT/ca.key and $OUT/ca.crt"
fi

# --- Leaf, signed by the CA (397 days, serverAuth) -------------------
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM
cat > "$WORK/leaf_ext.cnf" <<EOF
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=$SAN
EOF
echo "Issuing leaf for SAN $SAN (RSA 2048, SHA-256, 397 days)..."
openssl req -newkey rsa:2048 -keyout "$OUT/key.pem" \
    -out "$WORK/leaf.csr" -nodes -subj "/CN=$HOSTNAME"
openssl x509 -req -in "$WORK/leaf.csr" \
    -CA "$OUT/ca.crt" -CAkey "$OUT/ca.key" \
    -set_serial "0x$(openssl rand -hex 8)" \
    -out "$OUT/cert.pem" -days 397 -sha256 \
    -extfile "$WORK/leaf_ext.cnf"
echo "Wrote $OUT/cert.pem and $OUT/key.pem"

# --- Verify + show what the phone needs ------------------------------
echo
openssl verify -CAfile "$OUT/ca.crt" "$OUT/cert.pem"
openssl x509 -in "$OUT/cert.pem" -noout -subject -issuer -dates \
    -ext subjectAltName,extendedKeyUsage
echo
echo "NEXT STEPS (iPhone):"
echo "  1. Send ONLY $OUT/ca.crt to the iPhone (AirDrop/email)."
echo "     NEVER send $OUT/ca.key — it signs certificates."
echo "  2. On the iPhone tap the file, then"
echo "     Settings > Profile Downloaded > Install."
echo "  3. Settings > General > About > Certificate Trust Settings:"
echo "     switch ON 'Wall-Marker Local CA'."
echo "  4. Open https://$LAN_IP:8443/viewer (or https://192.168.137.1:8443/viewer"
echo "     on the Windows hotspot) — no certificate warning."
echo "  5. Start the server with --https --cert $OUT/cert.pem --key $OUT/key.pem"
