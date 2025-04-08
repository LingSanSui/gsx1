#!/bin/bash

# A basic container entrypoint to our build image that initializes a TMate session, using our
# own self-hosted tmate instance. This should give us most of what fly hijack offered us --
# an easy way to access the shell of a running build and poke around.

# I'd rather not be threading the private key through, but it's the best way
# IMO to manage the Git SSH access, as mounting the socket is problematic when the container
# is running as a user not present in the host's passwd file.
# IMO I don't think our builds running as root is a big deal but I'm sure there is a reason for it
mkdir -p ~/.ssh
eval "$(ssh-agent -s)"
ssh-add ~/host_ssh/id_rsa
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Copy the mounted docker creds to the home dir and chown it. I'm not
# allowed to idmap it for reasons I may figure out later.
mkdir -p "/home/$(whoami)/.docker"
sudo cp /docker_config.json "/home/$(whoami)/.docker/config.json"
sudo chown "$(whoami)": -Rf "/home/$(whoami)/.docker"

# Set it up to point out our self-hosted tmate service.
cat <<EOCONF | tee "$HOME/.tmate.conf"
set -g tmate-server-host tmate.gensyn.ai
set -g tmate-server-port 2200
set -g tmate-server-rsa-fingerprint SHA256:6nz4HyLchzIwSOKFPcX27DDYplGgS9luDitD5xDrPu4
set -g tmate-server-ed25519-fingerprint SHA256:cxBLDS/URGOG51NqNV8Ugn48rJ9WZcTILy65qjFPhTg
EOCONF

tmate -S /tmp/tmate new-session -d                              # Launch tmate in a headless mode

if timeout 30 tmate -S /tmp/tmate wait tmate-ready; then        # Wait for a tmate connection.
    connection=$(tmate -S /tmp/tmate display -p '#{tmate_ssh}') # Prints the SSH connection string if there wasn't a timeout
    printf '\033[32;1mPlease use the connection string below to SSH into the build container:\033[0m\n'
    printf '\033[32;1m############################################\033[0m\n\n'
    printf '\033[32;1m%s\033[0m\n\n' "$connection"
    printf '\033[32;1m############################################\033[0m\n'
else
    pkill tmate
    printf '\033[31;1mCould not establish tmate connection.\033[0m\n'
fi

printf '\n\n\033[33;1mBeginning Buildkite Step\033[0m\n\n'

sudo chmod u-s /usr/bin/new[gu]idmap
sudo setcap cap_setuid+eip /usr/bin/newuidmap
sudo setcap cap_setgid+eip /usr/bin/newgidmap

exec /bin/bash -c "$@"
