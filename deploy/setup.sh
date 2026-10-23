#!/usr/bin/env bash
# One-shot setup for an OCI Ubuntu instance (22.04 / 24.04, ARM Ampere or x86).
#
#   scp -r bg-remover ubuntu@<ip>:~
#   ssh ubuntu@<ip>
#   sudo bash ~/bg-remover/deploy/setup.sh               # serves on http://<ip>
#   sudo DOMAIN=bg.example.com bash ~/bg-remover/deploy/setup.sh   # HTTPS via Let's Encrypt
#
# Re-run it after copying new code; it is idempotent.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
APP=/opt/bg-remover
DOMAIN="${DOMAIN:-}"

apt-get update -y
apt-get install -y python3 python3-venv python3-pip rsync debian-keyring debian-archive-keyring apt-transport-https curl gnupg

# Caddy (reverse proxy + automatic HTTPS)
if ! command -v caddy >/dev/null; then
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -y
  apt-get install -y caddy
fi

id bgremover >/dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin bgremover
mkdir -p "$APP/models"
rsync -a --delete --exclude .venv --exclude models --exclude __pycache__ "$SRC/" "$APP/"

python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip
"$APP/.venv/bin/pip" install -q -r "$APP/requirements.txt"
chown -R bgremover:bgremover "$APP"

cp "$APP/deploy/bg-remover.service" /etc/systemd/system/bg-remover.service
systemctl daemon-reload
systemctl enable bg-remover
systemctl restart bg-remover

SITE="${DOMAIN:-:80}"
cat > /etc/caddy/Caddyfile <<EOF
$SITE {
	request_body {
		max_size 26MB
	}
	encode zstd gzip
	reverse_proxy 127.0.0.1:8000 {
		transport http {
			read_timeout 5m
		}
	}
}
EOF
systemctl reload caddy || systemctl restart caddy

# OCI's Ubuntu images ship iptables rules that drop everything except SSH.
for port in 80 443; do
  iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null || iptables -I INPUT 5 -p tcp --dport "$port" -j ACCEPT
done
command -v netfilter-persistent >/dev/null && netfilter-persistent save

echo
echo "Done. The first start downloads the model (~1 GB); follow it with:"
echo "  journalctl -u bg-remover -f"
echo "Also open ports 80/443 in the VCN Security List (OCI console) if you haven't."
