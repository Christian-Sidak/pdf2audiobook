#!/bin/bash
# Vast.ai control plane for remote stage-5 renders.
#   up                       find/start/create + provision the box
#   push <book> <voice>      sync code, voice (+sidecars), artifacts/<book>
#   render <book> <voice> [stages] [--no-tunnel]   launch remote render
#   status <book>            progress, rate, ETA, cost
#   log                      tail remote render log
#   pull <book>              bring back narration + segments + asr cache
#   ssh | stop | destroy
# Instance is discovered by label (never hardcoded); api key is read by the
# vastai CLI itself from ~/.config/vastai/vast_api_key.
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root

LABEL=pdf2audiobook-render
KEY=~/.ssh/pdf2audiobook_vast
REMOTE=/workspace/pdf2audiobook
STATE=scripts/vast/.vast_instance
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
QUERY='gpu_name=RTX_4090 num_gpus=1 rentable=true verified=true inet_down>300 inet_up>200 disk_space>60 cuda_vers>=12.1 reliability>0.98'

_json() { python3 -c "import json,sys; $1" ; }

_probe() { ssh -q -p "$2" -i "$KEY" -o IdentitiesOnly=yes \
       -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 "root@$1" true 2>/dev/null; }

_resolve() {  # populate ID HOST PORT from state file or vastai; tries BOTH
              # the vast proxy route and the direct host route (proxy is
              # sometimes dead on otherwise-healthy instances)
  if [ -f "$STATE" ]; then read -r ID HOST PORT < "$STATE"; fi
  if [ -n "${ID:-}" ] && _probe "${HOST:-x}" "${PORT:-0}"; then return 0; fi
  local row
  row=$(vastai show instances --raw | _json "
rows=[r for r in json.load(sys.stdin) if r.get('label')=='$LABEL'];
r=rows[0] if rows else {};
ports=(r.get('ports') or {}).get('22/tcp') or [];
direct=ports[0].get('HostPort','') if ports else '';
print(r.get('id',''), r.get('ssh_host',''), r.get('ssh_port',''),
      r.get('public_ipaddr',''), direct)")
  local PHOST PPORT DHOST DPORT
  read -r ID PHOST PPORT DHOST DPORT <<< "$row"
  [ -n "$ID" ] || return 1
  if [ -n "$PPORT" ] && _probe "$PHOST" "$PPORT"; then HOST=$PHOST; PORT=$PPORT
  elif [ -n "$DPORT" ] && _probe "$DHOST" "$DPORT"; then HOST=$DHOST; PORT=$DPORT
  else HOST=${PHOST:-$DHOST}; PORT=${PPORT:-$DPORT}; fi
  echo "$ID $HOST $PORT" > "$STATE"
}

r() {  # run a command on the box, MOTD filtered
  ssh -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "root@$HOST" "$@" \
      2>/dev/null | grep -vE 'Welcome to vast|Have fun'
}

_tar_push() { tar czf - "$@" | ssh -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes \
      -o StrictHostKeyChecking=accept-new "root@$HOST" "mkdir -p $REMOTE && tar xzf - -C $REMOTE"; }

cmd=${1:?usage: manage.sh up|push|render|status|log|pull|ssh|stop|destroy}; shift || true

case "$cmd" in
up)
  [ -f "$KEY" ] || ssh-keygen -t ed25519 -N '' -f "$KEY" -C pdf2audiobook-vast
  _resolve || true
  if [ -n "${ID:-}" ]; then
    st=$(vastai show instance "$ID" --raw | _json "print(json.load(sys.stdin).get('actual_status',''))")
    if [ "$st" = "offline" ]; then
      echo "labeled instance $ID is offline (dead host) — destroying and creating fresh"
      echo y | vastai destroy instance "$ID" || true
      rm -f "$STATE"; ID=""
    elif [ "$st" != "running" ]; then
      echo "starting stopped instance $ID"; vastai start instance "$ID"
    fi
  fi
  if [ -z "${ID:-}" ]; then
    echo "searching offers..."
    OFFER=$(vastai search offers "$QUERY" -o 'dph+' --raw | _json "
o=json.load(sys.stdin); print(o[0]['id'] if o else '')")
    [ -n "$OFFER" ] || { echo "no offers matched"; exit 1; }
    echo "creating instance from offer $OFFER"
    NEW=$(vastai create instance "$OFFER" --image "$IMAGE" --disk 60 --ssh --direct \
          --label "$LABEL" --raw | _json "print(json.load(sys.stdin).get('new_contract',''))")
    echo "instance $NEW created"
    vastai attach ssh "$NEW" "$(cat "$KEY.pub")" || true
  fi
  echo "waiting for ssh (image pull can take 15-25 min on first boot)..."
  for i in $(seq 1 90); do
    ID="" HOST="" PORT=""; _resolve
    [ -n "${ID:-}" ] && [ -n "${HOST:-}" ] && r true 2>/dev/null && break
    sleep 20
  done
  r true || { echo "ssh never came up"; exit 1; }
  echo "instance $ID at $HOST:$PORT — syncing code + provisioning"
  _tar_push pipeline evals export scripts main.py config.yaml requirements-cuda.txt
  r "bash $REMOTE/scripts/vast/bootstrap.sh"
  echo "UP: bootstrap complete"
  ;;

push)
  book=${1:?book}; voice=${2:?voice}
  _resolve; [ -n "$ID" ] || { echo "no instance; run up first"; exit 1; }
  _tar_push pipeline evals export scripts main.py config.yaml requirements-cuda.txt
  files=("voices/$voice.wav")
  for s in roomtone.wav ref_text.txt; do [ -f "voices/$voice.$s" ] && files+=("voices/$voice.$s"); done
  _tar_push "${files[@]}"
  [ -d "evals/.cache/asr" ] && _tar_push evals/.cache/asr
  tar czf - "artifacts/$book" | ssh -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes \
      -o StrictHostKeyChecking=accept-new "root@$HOST" "tar xzf - -C $REMOTE"
  r "mkdir -p $REMOTE/library && touch '$REMOTE/library/$book.pdf'"
  lsha=$(shasum -a 256 "voices/$voice.wav" | cut -d' ' -f1)
  rsha=$(r "sha256sum '$REMOTE/voices/$voice.wav'" | cut -d' ' -f1)
  [ "$lsha" = "$rsha" ] || { echo "VOICE SHA MISMATCH — aborting (cache key would diverge)"; exit 1; }
  echo "PUSH OK: voice sha verified ($lsha)"
  ;;

render)
  book=${1:?book}; voice=${2:?voice}; stages=${3:-5}
  _resolve; [ -n "$ID" ] || { echo "no instance; run up first"; exit 1; }
  if [ "${4:-}" != "--no-tunnel" ] && [ "${3:-}" != "--no-tunnel" ]; then
    ssh -f -N -R 11434:localhost:11434 -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes \
        -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "root@$HOST" \
        && pgrep -f "ssh -f -N -R 11434.*$HOST" | head -1 > scripts/vast/.tunnel_pid \
        && echo "ollama tunnel up" || echo "tunnel failed (adjudication degrades gracefully)"
  fi
  r "cd $REMOTE && rm -f 'artifacts/$book/RENDER_DONE' && mkdir -p artifacts && \
     nohup bash scripts/vast/render_job.sh '$book' '$voice' '$stages' \
       > artifacts/render.log 2>&1 & echo \$! > artifacts/render.pid"
  echo "render launched; waiting for first signs of life..."
  for i in $(seq 1 60); do
    line=$(r "grep -m1 -E 'rendered|render|Traceback|Error' $REMOTE/artifacts/render.log 2>/dev/null | head -1" || true)
    [ -n "$line" ] && { echo "$line"; break; }
    sleep 5
  done
  nohup bash "scripts/vast/manage.sh" _watchdog "$book" > scripts/vast/.watchdog.log 2>&1 &
  echo "watchdog spawned (auto-stops instance on RENDER_DONE)"
  ;;

_watchdog)
  book=${1:?book}
  _resolve
  # DOCTRINE (learned 2026-08-10, three killed renders): NEVER stop the box on
  # unreachability — ssh from this Mac to vast hosts drops for >1h stretches
  # while renders run fine, so silence is not evidence of waste. Stop ONLY on
  # the DONE sentinel, or at a hard 12h deadline as runaway protection.
  deadline=$(( $(date +%s) + ${WATCHDOG_HOURS:-12}*3600 ))
  while true; do
    sleep 300
    if r "test -f '$REMOTE/artifacts/$book/RENDER_DONE'" 2>/dev/null; then
      echo "$(date) RENDER_DONE — stopping instance $ID"
      vastai stop instance "$ID" || true
      [ -f scripts/vast/.tunnel_pid ] && kill "$(cat scripts/vast/.tunnel_pid)" 2>/dev/null || true
      osascript -e 'display notification "Remote render finished; instance stopped. Run manage.sh pull." with title "vast.ai"' 2>/dev/null || true
      break
    fi
    r true 2>/dev/null || _resolve || true
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "$(date) 12h hard deadline — stopping instance $ID"
      vastai stop instance "$ID" || true
      break
    fi
  done
  ;;

status)
  book=${1:?book}
  _resolve; [ -n "$ID" ] || { echo "no instance"; exit 1; }
  total=$(python3 -c "import json; print(len(json.load(open('artifacts/$book/04_narration.json'))['segments']))" 2>/dev/null || echo "?")
  n=$(r "ls $REMOTE/artifacts/$book/05_render/segments 2>/dev/null | wc -l" | tr -d ' ')
  recent=$(r "find $REMOTE/artifacts/$book/05_render/segments -mmin -30 2>/dev/null | wc -l" | tr -d ' ')
  rate=$((recent * 2))
  done_flag=$(r "test -f '$REMOTE/artifacts/$book/RENDER_DONE' && echo DONE" || true)
  dph=$(vastai show instance "$ID" --raw | _json "r=json.load(sys.stdin); print(r.get('dph_total','?'))")
  echo "$book: $n / $total takes  rate ${rate}/hr  ${done_flag:-rendering}  (\$${dph}/hr)"
  if [ "$rate" -gt 0 ] && [ "$total" != "?" ]; then
    python3 -c "print(f'ETA ~{(($total-$n)/$rate):.1f}h')"
  fi
  r "tail -2 $REMOTE/artifacts/render.log" || true
  ;;

log) _resolve; r "tail -60 $REMOTE/artifacts/render.log" ;;

pull)
  book=${1:?book}
  _resolve; [ -n "$ID" ] || { echo "no instance"; exit 1; }
  ssh -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
      "root@$HOST" "cd $REMOTE/artifacts/$book && tar czf - 04_narration.json 05_render" \
      | tar xzf - -C "artifacts/$book"
  ssh -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
      "root@$HOST" "cd $REMOTE/evals/.cache && tar czf - asr" \
      | tar xzf - -C evals/.cache 2>/dev/null || true
  echo "PULL OK. Verify locally:"
  echo "  python3 main.py status $book"
  echo "  python3 main.py build 'library/<book pdf>' --stages 5 ... (expect all cache hits)"
  read -r -p "stop instance now? [Y/n] " a
  [ "${a:-Y}" != "n" ] && vastai stop instance "$ID"
  ;;

ssh) _resolve; exec ssh -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes "root@$HOST" ;;
stop) _resolve; vastai stop instance "$ID" ;;
destroy) _resolve; echo y | vastai destroy instance "$ID" && rm -f "$STATE" ;;
*) echo "unknown verb: $cmd"; exit 1 ;;
esac
