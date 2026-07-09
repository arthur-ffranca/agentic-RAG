# Creative Video Assets

This folder stores generated image and video assets for the project demo.

## Nano Banana MCP

Local MCP server:

```text
mcp_nano_server.py
```

Required environment variable in the project `.env`:

```env
ACEDATACLOUD_API_TOKEN=your_token_here
```

Example Codex MCP config:

```toml
[mcp_servers.nano_banana_creative]
command = "python"
args = ["C:\\Users\\otavi\\OneDrive\\Desktop\\path-complete\\mcp_nano_server.py"]
```

Available tools:

```text
generate_image_asset
generate_project_cover
create_video_prompt
```

## Veo MCP

Local MCP server:

```text
mcp_veo_server.py
```

Example Codex MCP config:

```toml
[mcp_servers.veo_creative_video]
command = "python"
args = ["C:\\Users\\otavi\\OneDrive\\Desktop\\path-complete\\mcp_veo_server.py"]
```

Available tools:

```text
generate_video_from_prompt
generate_project_demo_clip
generate_video_from_image
```

Video generation can take several minutes. Use `timeout_seconds` when calling
the tools if the request needs more time.
