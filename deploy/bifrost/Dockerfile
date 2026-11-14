# Minimal public Bifrost AI Gateway deployment
# Uses official Maxim AI Bifrost image (https://docs.getbifrost.ai)
FROM maximhq/bifrost:latest

# Copy declarative config into container data directory
COPY config.json /app/data/config.json

# Environment variables for container web service
# Render, Railway, and Cloud platforms require binding to 0.0.0.0 and port 10000
ENV APP_HOST=0.0.0.0
ENV APP_PORT=10000
ENV PORT=10000
ENV LOG_LEVEL=info
ENV LOG_STYLE=json

EXPOSE 10000
