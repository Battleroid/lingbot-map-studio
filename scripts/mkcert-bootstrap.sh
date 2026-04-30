#!/usr/bin/env bash
# scripts/mkcert-bootstrap.sh
#
# One-shot bootstrap for HTTPS-via-Caddy: installs mkcert if missing,
# trusts the local root CA, generates a cert pair into caddy/certs/,
# and copies the root CA next to it so Caddy can serve it over plain
# HTTP (the phone has to download + trust the CA *before* it'll accept
# the studio's HTTPS cert, classic chicken-and-egg).
#
# Idempotent: safe to re-run. A second invocation reuses the existing
# mkcert binary + root CA and just overwrites the cert pair (cheap).
#
# Env knobs:
#   STUDIO_HOSTNAME — friendly DNS name to bake into the cert
#                     (default: studio.local). The certs always cover
#                     localhost + 127.0.0.1 + the auto-detected LAN IP
#                     too, so accessing by IP works regardless.
#   STUDIO_LAN_IP   — override the LAN IP (otherwise auto-detected from
#                     `ip addr` / `ipconfig`).
#   CERTS_DIR       — output dir (default: caddy/certs).

set -euo pipefail

CERTS_DIR="${CERTS_DIR:-caddy/certs}"
HOSTNAME_DEFAULT="${STUDIO_HOSTNAME:-studio.local}"

# ── colour helpers ────────────────────────────────────────────────
red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
plus()   { printf '\033[36m[+] %s\033[0m\n' "$*"; }

# ── 1. ensure mkcert is present ────────────────────────────────────
ensure_mkcert() {
    if command -v mkcert >/dev/null 2>&1; then
        return 0
    fi
    plus "mkcert not found — installing"

    local uname_s
    uname_s=$(uname -s)

    case "$uname_s" in
        Linux)
            # Try distro packages first (Debian/Ubuntu 22+ ship it). Fall
            # back to the upstream GitHub release binary if apt doesn't
            # know about mkcert or we're on a non-apt distro.
            if command -v apt-get >/dev/null 2>&1; then
                if [[ $EUID -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
                    red "    need root or sudo to install libnss3-tools"
                    exit 1
                fi
                local SUDO=""
                [[ $EUID -ne 0 ]] && SUDO="sudo"
                $SUDO apt-get update -y >/dev/null
                $SUDO apt-get install -y libnss3-tools ca-certificates curl >/dev/null
                if ! $SUDO apt-get install -y mkcert >/dev/null 2>&1; then
                    install_mkcert_from_github "$SUDO"
                fi
            elif command -v dnf >/dev/null 2>&1; then
                local SUDO=""
                [[ $EUID -ne 0 ]] && SUDO="sudo"
                $SUDO dnf install -y nss-tools curl >/dev/null
                install_mkcert_from_github "$SUDO"
            else
                # Unknown distro — try a sudo-less install to /usr/local.
                install_mkcert_from_github sudo
            fi
            ;;
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                brew install mkcert nss
            else
                red "    need Homebrew to install mkcert on macOS."
                red "    install brew (https://brew.sh) and re-run, or"
                red "    install mkcert manually from"
                red "    https://github.com/FiloSottile/mkcert/releases"
                exit 1
            fi
            ;;
        *)
            red "    unsupported OS: $uname_s"
            red "    install mkcert manually from"
            red "    https://github.com/FiloSottile/mkcert/releases"
            exit 1
            ;;
    esac
}

# ── 1b. ensure qrencode is present (best-effort, no-op on failure) ─
# Used by the Makefile to print a scannable QR code for the rootCA URL
# so the user can point their phone camera at the terminal instead of
# typing the LAN IP. Optional — if the install fails (e.g. the host is
# offline or on an exotic distro) the Makefile falls back to printing
# the URL as plain text.
ensure_qrencode() {
    if command -v qrencode >/dev/null 2>&1; then
        return 0
    fi
    plus "qrencode not found — attempting install (best-effort)"
    local uname_s
    uname_s=$(uname -s)
    case "$uname_s" in
        Linux)
            local SUDO=""
            [[ $EUID -ne 0 ]] && SUDO="sudo"
            if command -v apt-get >/dev/null 2>&1; then
                $SUDO apt-get install -y qrencode >/dev/null 2>&1 || \
                    yellow "[!] couldn't install qrencode; QR will be skipped"
            elif command -v dnf >/dev/null 2>&1; then
                $SUDO dnf install -y qrencode >/dev/null 2>&1 || \
                    yellow "[!] couldn't install qrencode; QR will be skipped"
            else
                yellow "[!] no apt/dnf — install qrencode yourself for the QR"
            fi
            ;;
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                brew install qrencode >/dev/null 2>&1 || true
            fi
            ;;
    esac
}

install_mkcert_from_github() {
    local SUDO="${1:-}"
    local mkarch
    case "$(uname -m)" in
        x86_64|amd64)         mkarch=amd64 ;;
        aarch64|arm64)        mkarch=arm64 ;;
        armv7l|armhf)         mkarch=arm   ;;
        *) red "    unknown arch $(uname -m)"; exit 1 ;;
    esac
    local url="https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-${mkarch}"
    plus "downloading mkcert v1.4.4 (${mkarch})"
    $SUDO curl -fsSL -o /usr/local/bin/mkcert "$url"
    $SUDO chmod +x /usr/local/bin/mkcert
}

# ── 2. trust the local root CA ─────────────────────────────────────
ensure_root_ca() {
    plus "installing local root CA into the system trust store"
    plus "(this may prompt for sudo on Linux — needed once per host)"
    if ! mkcert -install; then
        yellow "[!] mkcert -install failed — host browsers may not auto-trust"
        yellow "    the cert. Phone trust still works via the rootCA.pem download."
    fi
}

# ── 3. detect the host's LAN IP ────────────────────────────────────
detect_lan_ip() {
    local ip="${STUDIO_LAN_IP:-}"
    if [[ -n "$ip" ]]; then
        echo "$ip"
        return
    fi
    # Linux: pick the first non-loopback non-docker non-bridge IPv4.
    if command -v ip >/dev/null 2>&1; then
        ip=$(ip -4 -o addr show 2>/dev/null \
            | awk '$2 !~ /^(lo|docker|br-|veth|virbr|tun|tap)/ {split($4, a, "/"); print a[1]; exit}')
    fi
    # macOS: try en0 then en1.
    if [[ -z "$ip" ]] && command -v ipconfig >/dev/null 2>&1; then
        ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
    fi
    echo "$ip"
}

# ── 4. generate the cert pair ──────────────────────────────────────
generate_cert() {
    local lan_ip="$1"
    mkdir -p "$CERTS_DIR"
    local sans=("$HOSTNAME_DEFAULT" localhost 127.0.0.1)
    if [[ -n "$lan_ip" ]]; then
        sans+=("$lan_ip")
    fi
    plus "generating cert for: ${sans[*]}"
    mkcert -cert-file "$CERTS_DIR/cert.pem" \
           -key-file  "$CERTS_DIR/key.pem" \
           "${sans[@]}" >/dev/null
}

# ── 5. copy the root CA next to the cert pair ──────────────────────
copy_root_ca() {
    local caroot
    caroot=$(mkcert -CAROOT)
    if [[ ! -f "$caroot/rootCA.pem" ]]; then
        red "    expected root CA at $caroot/rootCA.pem, not found"
        exit 1
    fi
    cp "$caroot/rootCA.pem" "$CERTS_DIR/rootCA.pem"
    chmod 644 "$CERTS_DIR/rootCA.pem"
    # Also expose the same root CA at a `.crt` path. Android's
    # "Install a certificate → CA certificate" picker filters its file
    # browser by extension and hides anything that isn't `.crt` /
    # `.cer` / `.pkcs12`, so a plain `.pem` won't show up — even
    # though the file format is byte-for-byte the same. Caddy serves
    # the .crt URL with `Content-Type: application/x-x509-ca-cert`
    # which is what Android's install-on-tap path expects.
    cp "$caroot/rootCA.pem" "$CERTS_DIR/rootCA.crt"
    chmod 644 "$CERTS_DIR/rootCA.crt"
}

# ── main ───────────────────────────────────────────────────────────
ensure_mkcert
ensure_qrencode
ensure_root_ca

LAN_IP=$(detect_lan_ip)
generate_cert "$LAN_IP"
copy_root_ca

# Stash the detected LAN IP next to the certs so the Makefile can pick
# it up for its post-up summary (without re-running detection).
{
    [[ -n "$LAN_IP" ]] && echo "STUDIO_LAN_IP=$LAN_IP"
    echo "STUDIO_HOSTNAME=$HOSTNAME_DEFAULT"
} > "$CERTS_DIR/.env.bootstrap"

green ""
green "[+] cert pair written to $CERTS_DIR/{cert,key}.pem"
green "[+] root CA copied to    $CERTS_DIR/rootCA.pem"
if [[ -n "$LAN_IP" ]]; then
    green "[+] detected LAN IP:     $LAN_IP"
fi
green "[+] friendly hostname:   $HOSTNAME_DEFAULT"
