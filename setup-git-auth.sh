#!/usr/bin/env bash
# Repeatable SSH re-auth for THIS repo to a personal GitHub account,
# fully isolated from the default github.com identity (Luminary / saakaarbh).
#
#   - dedicated ssh key (SSH_KEY)
#   - a fake Host alias (SSH_ALIAS) in ~/.ssh/config that forces that key
#   - this repo's origin points at the alias; nothing else does
#
# Re-run anytime auth drifts:  ./setup-git-auth.sh
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-git-auth.config}"
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG"

case "${PERSONAL_EMAIL:-}" in
  *luminarycloud.com|""|CHANGE_ME@example.com)
    echo "ERROR: set a personal PERSONAL_EMAIL in $CONFIG"; exit 1 ;;
esac
: "${GITHUB_USER:?set GITHUB_USER}"; : "${REPO:?set REPO}"
: "${SSH_KEY:?set SSH_KEY}"; : "${SSH_ALIAS:?set SSH_ALIAS}"

SSH_CONFIG="$HOME/.ssh/config"

# --- 1. identity (local only) ---
git config --local user.name  "$PERSONAL_NAME"
git config --local user.email "$PERSONAL_EMAIL"

# --- 2. dedicated key (create if missing; never overwrite) ---
if [ ! -f "$SSH_KEY" ]; then
  ssh-keygen -t ed25519 -C "$PERSONAL_EMAIL" -f "$SSH_KEY" -N "" >/dev/null
  echo "Created new key: $SSH_KEY"
else
  echo "Reusing existing key: $SSH_KEY"
fi

# --- 3. ~/.ssh/config alias block (idempotent, marker-delimited) ---
mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"; touch "$SSH_CONFIG"; chmod 600 "$SSH_CONFIG"
BEGIN="# >>> pfit-claude personal ($SSH_ALIAS) >>>"
END="# <<< pfit-claude personal ($SSH_ALIAS) <<<"
# strip any prior block, then append a fresh one
tmp="$(mktemp)"
awk -v b="$BEGIN" -v e="$END" '
  $0==b {skip=1} !skip {print} $0==e {skip=0}
' "$SSH_CONFIG" > "$tmp"
{
  cat "$tmp"
  printf '%s\n' "$BEGIN"
  printf 'Host %s\n' "$SSH_ALIAS"
  printf '    HostName github.com\n'
  printf '    User git\n'
  printf '    IdentityFile %s\n' "$SSH_KEY"
  printf '    IdentitiesOnly yes\n'
  printf '%s\n' "$END"
} > "$SSH_CONFIG"
rm -f "$tmp"
chmod 600 "$SSH_CONFIG"

# --- 4. this repo's remote uses the alias (nothing else does) ---
git remote set-url origin "git@${SSH_ALIAS}:${REPO}.git"

# --- 5. drop any leftover HTTPS/PAT cred config from earlier attempts ---
git config --local --unset-all credential.helper 2>/dev/null || true
rm -f "$(git rev-parse --git-dir)/personal.credentials"

# --- 6. verify ---
echo "----------------------------------------"
echo "Identity : $(git config --local user.name) <$(git config --local user.email)>"
echo "Remote   : $(git remote get-url origin)"
echo "Key      : $SSH_KEY"
echo "Alias    : $SSH_ALIAS -> github.com (IdentitiesOnly)"
echo -n "SSH auth : "
if out="$(ssh -T -o StrictHostKeyChecking=accept-new "git@${SSH_ALIAS}" 2>&1)"; then :; fi
echo "$out"
echo "----------------------------------------"
if echo "$out" | grep -qi "Hi ${GITHUB_USER}!"; then
  echo "OK - authenticated as ${GITHUB_USER}. You can now: git push origin <branch>"
else
  echo "ACTION NEEDED: add this PUBLIC key to the ${GITHUB_USER} account"
  echo "  (github.com -> Settings -> SSH and GPG keys -> New SSH key), then re-run:"
  echo "----- copy the line below -----"
  cat "${SSH_KEY}.pub"
  echo "-------------------------------"
fi
