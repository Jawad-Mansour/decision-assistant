#!/bin/sh
# Render nginx config from template at container start.
# We DELIBERATELY keep the template outside /etc/nginx/templates/ so the nginx
# base image's built-in 20-envsubst-on-templates.sh does NOT run on it (that
# one substitutes every $VAR, including nginx's own $host, $remote_addr, ...).
# Instead we run envsubst ourselves with an explicit whitelist.
set -eu

: "${PORT:=80}"
: "${BACKEND_HOST:=backend}"
: "${BACKEND_PORT:=8000}"

export PORT BACKEND_HOST BACKEND_PORT

envsubst '${PORT} ${BACKEND_HOST} ${BACKEND_PORT}' \
    < /etc/nginx/conf.template/default.conf.template \
    > /etc/nginx/conf.d/default.conf
