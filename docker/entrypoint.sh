#!/bin/bash
set -euo pipefail

mkdir -p /root/.ssh
echo "${SSH_PUBKEY:?}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

if service ssh start; then
    echo "SSHD started successfully."
else
    echo "ERROR: Failed to start SSHD via service command."
fi

tail -f /dev/null
