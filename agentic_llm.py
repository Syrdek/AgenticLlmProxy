import asyncio
import datetime
import json
import logging
from typing import Any, Callable

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.schema import StreamEvent
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient


def browse_obj(obj, *path):
    """
    Hepls to browse object for data.
    :param obj: The object to browse.
    :param path: The chain of attributes to browse.
    :return: The attribute at the end of the chain.
    """
    if obj is None:
        return None

    if len(path) == 0:
        return obj

    args = list(path)
    arg = args.pop(0)

    if isinstance(obj, dict):
        return browse_obj(obj.get(arg), *args)

    if isinstance(obj, list):
        idx = int(arg)
        if len(obj) > idx:
            return browse_obj(obj[idx], *args)
        return None

    if hasattr(obj, arg):
        return browse_obj(getattr(obj, arg), *args)


class StreamEventParser(object):
    """
    Parses events, and stores information.
    """

    def __init__(self):
        self.model = None
        self.steps = []
        self.sources = []
        self.received_chat_evts = False

    def __parse_chain_event(self, chain_event: StreamEvent) -> dict[str, Any]:
        """
        Parses a chain event.
        :param chain_event: The event to parse.
        :return: The information extracted from the event.
        """
        data = browse_obj(chain_event, "data")
        chunk = browse_obj(data, "chunk")
        output = browse_obj(data, "output")
        message = chunk if chunk else output
        # In on_chain_stream, message can be inside a sub-list named 'messages'
        message = browse_obj(message, "messages", -1) or message
        content = browse_obj(message, "content") or ''
        model = browse_obj(chain_event, "metadata", "ls_model_name")
        thinking = browse_obj(message, "additional_kwargs", "reasoning_content")
        tool_calls = browse_obj(data, "tool_calls")
        metadata = browse_obj(chain_event, "response_metadata") or browse_obj(message, "response_metadata")

        if model:
            self.model = model
        else:
            model = self.model

        result = {
            "model": model,
            "message": {
                "role": "assistant",
                "content": content
            },
            "chain": {
                "step": len(self.steps),
                "sources": self.sources
            }
        }

        if metadata:
            result.update(metadata)

        if thinking:
            result["message"]["thinking"] = thinking

        if tool_calls:
            result["chain"]["tool_calls"] = tool_calls

        result["created_at"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        result["done"] = False

        return result

    def parse(self, stream_event: StreamEvent) -> dict[str, Any] | None:
        """
        Parses a processing chain event.
        :param stream_event: The event to parse.
        :return: A dict resuming the event. Might be None if the event had no valuable information.
        """
        evt_type = stream_event.get("event")

        if evt_type == "on_chat_model_end":
            self.received_chat_evts = True
            # evt = self.parse_chat_model_event(stream_event)
            # self.steps.append(evt)

        if evt_type == "on_chain_stream":
            # Compatibility with versions not sending 'on_chat_model_*' events
            name = browse_obj(stream_event, "name")
            if name == "model":
                evt = self.__parse_chain_event(stream_event)
                self.steps.append(evt)
                # If we has 'on_chat_model_*' evts, we don't want to send this chunk.
                return evt if not self.received_chat_evts else None

        if evt_type == "on_chat_model_stream":
            self.received_chat_evts = True
            return self.__parse_chain_event(stream_event)

        if evt_type == "on_tool_end":
            output = browse_obj(stream_event, "data", "output", "content")
            try:
                json_tool = json.loads(output)
                source = browse_obj(json_tool, "__mcp_metadata", "source")
                if source:
                    self.sources.append(source)
            except:
                logging.info("Tool contained no mcp metadata")

        return {}

    def end(self):
        return {
            "model": self.model,
            "message": {
                "role": "assistant",
                "content": ''
            },
            "chain": {
                "step": len(self.steps),
                "sources": self.sources
            },
            "done": True
        }

    def summarize(self):
        if not self.steps:
            return self.end()

        result = self.steps[-1].copy()
        result["done"] = True
        return result


def create_llm(config: dict[str, Any], **options) -> BaseChatModel:
    """
    Creates the LLM connection.
    :param config: The LLM configuration.
    :param options: The LLm creation options.
    :return: The created LLM.
    """
    provider = str(config.get("provider", "ollama")).lower()
    if "ollama" == provider:
        return ChatOllama(base_url=config.get("url", "http://localhost:11434"), **options)
    else:
        raise TypeError(f"Provider {provider} is not managed")


async def stream(llm: dict[str, Any],
                 agents: dict[str, Any],
                 request: dict[str, Any],
                 callback: Callable[[dict[str, Any]], None] = None):
    """
    Calls the distant LLM with the given tools.
    :param llm: A dict containing LLM configuration.
    :param agents: A dict of agents to use.
    :param request: The request to send to the LLM.
    :param callback: An optional callback that will be notified of the progress.
    :return: The final response of the LLM, after using agents if necessary.
    """
    llm = create_llm(llm, **request)

    client = MultiServerMCPClient(agents)

    tools = await client.get_tools()
    agent = create_agent(llm, tools)

    parser = StreamEventParser()
    async for evt in agent.astream_events(request):
        chunk = parser.parse(evt)
        if chunk:
            if callback:
                callback(chunk)

    if callback:
        callback(parser.end())

    return parser.summarize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open("config.json", "r") as f:
        conf = json.load(f)

    logging.warning(asyncio.run(stream(
        **conf,
        request={
            "model": "qwen3:30b-a3b",
            "messages": [{
                "role": "user",
                "content": "A quoi sert le fichier 'main.py' ?"
            }],
            "stream": True,
            "reasoning": True
        }, callback=logging.info)))
