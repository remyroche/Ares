#!/bin/bash

echo "🔍 Monitoring for GLM-5 API calls..."
echo "Looking for connections to api.z.ai"
echo ""
echo "Run this while using Claude Code, then ask Claude something."
echo "Press Ctrl+C to stop."
echo ""

# Monitor network connections to ZhipuAI API
sudo lsof -i -P | grep -i "api.z.ai" || echo "No active connections to api.z.ai yet"

# Alternative: Monitor with tcpdump (requires sudo)
# sudo tcpdump -i any -n host api.z.ai
