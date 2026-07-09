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

## MuAPI + Seedance MCP

Local MCP wrapper:

```text
mcp_muapi_server.py
```

Required environment variable in the project `.env`:

```env
MUAPI_API_KEY=your_key_here
```

The wrapper also accepts these fallback names for local experimentation:

```text
MUAPIAPP_API_KEY
MUAI_API_KEY
SEEDANCE_API_TOKEN
```

Final generated project video:

```text
creative_video/output/maritaca-hybrid-graph-agentic-rag-seedance.mp4
```

Generation details:

```text
provider: MuAPI
model: seedance-2
mode: image-to-video
duration: 5 seconds
cost: $1.25
```
