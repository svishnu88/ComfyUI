import os
import json
import glob
import urllib
from aiohttp import web
import aiohttp
import node
import websockets
import asyncio

routes = web.RouteTableDef()
web_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

@routes.get("/")
async def get_root(request):
    return web.FileResponse(os.path.join(web_root, "index.html"))

@routes.get("/users")
async def get_users(request):
    return web.json_response({"users": list(request.app["users"].values())})

@routes.get("/extensions")
async def get_extensions(request):
    files = glob.glob(os.path.join(glob.escape(web_root), 'extensions/**/*.js'), recursive=True)
    extensions = list(map(lambda f: "/" + os.path.relpath(f, web_root).replace("\\", "/"), files))

    for name, dir in node.EXTENSION_WEB_DIRS.items():
        files = glob.glob(os.path.join(glob.escape(dir), '**/*.js'), recursive=True)
        extensions.extend(list(map(lambda f: "/extensions/" + urllib.parse.quote(name) + "/" + os.path.relpath(f, dir).replace("\\", "/"), files)))

    return web.json_response(extensions)

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                text = await response.text()
                data = json.loads(text)
                return data
            else:
                return {'error': f'Failed to fetch data, status code: {response.status}'}

@routes.get("/object_info")
async def main(request):
    url = 'https://raw.githubusercontent.com/Kiruthika-V-G/comfyui-nodes/main/object_info.json'  # Replace with your endpoint
    data = await fetch_data(url)
    return web.json_response(data)

async def websocket_to_cloud():
    uri = "wss://b2a6225d81e01.notebooksc.jarvislabs.net//ws"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            print(f"Received from cloud: {data}")
            await websocket.send("Message from local server")


def create_app():
    app = web.Application()
    app.add_routes(routes)
    app.add_routes([web.static('/', web_root)])
    return app

if __name__ == "__main__":
    app = create_app()
    loop = asyncio.get_event_loop()
    loop.create_task(websocket_to_cloud())
    web.run_app(app, host="localhost", port=8080)
