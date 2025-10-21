import asyncio
import websockets
import json

async def main():
    async with websockets.connect("ws://10.112.148.127:8765") as ws:
        data = [1, 2, 3, 4, 5]
        await ws.send(json.dumps(data))
        resp = await ws.recv()
        print(resp)

asyncio.run(main())
