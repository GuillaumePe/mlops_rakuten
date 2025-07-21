#!/usr/bin/env bash
# wait-for-it.sh from https://github.com/vishnubob/wait-for-it
set -e
set -x

host="$1"
port="$2"
shift 2
cmd="$@"

until nc -z "$host" "$port"; do
  >&2 echo "Waiting for $host:$port to be available..."
  sleep 1
done

>&2 echo "$host:$port is up — executing command"
exec $cmd
