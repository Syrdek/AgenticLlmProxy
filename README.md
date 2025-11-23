# AgenticLlmProxy

A proxy that adds mcp capabilities to a distant LLM.

Responses streaming is supported if python >= 3.12.

# Configuration

Overview the file `config.json` file :
```json
{
  "llm": {
    "provider": "ollama",
    "url": "http://localhost:11434"
  },
  "proxy": {
    "host": "0.0.0.0",
    "port": 11435
  },
  "agents": {
    "filesystem": {
      "transport": "sse",
      "url": "http://localhost:48000/sse"
    }
  }
}
```

### llm section

Configures the LLM server to proxy requests to.
- provider : The distant LLM server type. Actually, only ollama is managed, see agentic_llm.create_llm to support other LLMs.
- url : The server URL (must expose a /api/chat path)

### proxy section

Configures the http interface exposed by the proxy.
Options here are given to the waitress server. You can add any waitress parameter here.
- host : The host to listen on.
- port : The port to listen on.

See https://docs.pylonsproject.org/projects/waitress/en/latest/arguments.html for more arguments.

Note : to call the agentic proxy, the url will be : `http://[host]:[port]/api/chat`

### agents section

Configures MCP agents available.
Agents configures here are in the langchain_mcp_adapters.MultiServerMCPClient format.
See https://docs.langchain.com/oss/python/langchain/mcp#use-mcp-tools for more informations.