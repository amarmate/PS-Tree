#!/bin/bash
set -euo pipefail

mkdir -p /root/.ssh
echo "${SSH_PUBKEY:?}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

service ssh start

tail -f /dev/null
