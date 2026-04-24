# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS build

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY frontend/ ./

# Leave empty so the browser uses same-origin relative URLs (/query, /health, ...)
# and nginx in this same container proxies them to the backend over the internal
# docker network. Override only if you want the browser to hit a different origin.
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

FROM nginx:1.27-alpine AS runtime

RUN apk add --no-cache gettext wget

COPY --from=build /app/dist /usr/share/nginx/html
COPY frontend/nginx.conf.template /etc/nginx/conf.template/default.conf.template
COPY frontend/docker-entrypoint.sh /docker-entrypoint.d/40-render-nginx-conf.sh

RUN chmod +x /docker-entrypoint.d/40-render-nginx-conf.sh \
    && rm -f /etc/nginx/conf.d/default.conf

ENV PORT=80 \
    BACKEND_HOST=backend \
    BACKEND_PORT=8000

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=5 \
    CMD wget -qO- http://127.0.0.1:${PORT}/ >/dev/null || exit 1
