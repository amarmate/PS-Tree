#!/bin/bash
set -euo pipefail

mkdir -p /root/.ssh
echo "${SSH_PUBKEY:?}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# Starte den SSH-Server im Vordergrund
exec /usr/sbin/sshd -D

tail -f /dev/null
