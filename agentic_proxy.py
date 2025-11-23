import asyncio
import json
import logging
from queue import Queue
from threading import Thread
from typing import Any, Generator

import flask
import requests

from agentic_llm import stream

app = flask.Flask("Agentic-Proxy")


def sync_stream(request: dict[str, Any], timeout: int|None = None) -> Generator[dict[str, Any]]:
    """
    Converts the async llm stream method using callbacks to a sync generator method
    :param request: The request to perform.
    :param timeout: A timeout used to stop request if delay between each message is too long.
    :return: A generator returning ollama format events.
    """
    q = Queue()
    job_done = object()

    def enqueue_event(x):
        logging.info(x)
        q.put(x)

    def task():
        asyncio.run(stream(llm=conf["llm"], agents=conf["agents"], request=request, callback=enqueue_event))
        q.put(job_done)

    Thread(target=task, args=(), daemon=True).start()

    while True:
        next_item = q.get(True)
        if next_item is job_done:
            break
        yield json.dumps(next_item) + "\n"


@app.route("/api/chat", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def ollama_default() -> flask.Response:
    """
    Proxies the request to the target LLM, and adds agents processing chain.
    :return: The response.
    """
    data = flask.request.get_json()

    if data.get("stream", True):
        return flask.Response(response=sync_stream(request=data), content_type="application/x-ndjson")

    result = asyncio.run(stream(llm=conf["llm"], agents=conf["agents"], request=data))
    return flask.Response(response=json.dumps(result), content_type="application/json")


@app.route('/', defaults={'path': ''}, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
@app.route('/<path:path>', methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def proxy(path: str) -> flask.Response:
    """
    Proxies the request with no changes to the target server.
    :param path: The called path.
    :return: The target response.
    """
    res = requests.request(method=flask.request.method,
                           url=flask.request.url.replace(flask.request.host_url, f'{conf["llm"]["url"]}/'),
                           headers={k: v for k, v in flask.request.headers if k.lower() != 'host'},
                           data=flask.request.get_data(),
                           cookies=flask.request.cookies,
                           allow_redirects=False,
                           )

    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(k, v) for k, v in res.raw.headers.items() if k.lower() not in excluded_headers]
    return flask.Response(res.content, res.status_code, headers)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with open("config.json", "r") as f:
        conf = json.load(f)

    proxy_conf = conf.get("proxy", {})
    if proxy_conf.get("debug", False):
        app.run(**proxy_conf)
    else:
        from waitress import serve
        serve(app, **proxy_conf)
