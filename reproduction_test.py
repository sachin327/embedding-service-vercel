import asyncio
import httpx
import time
import random

API_URL = "http://localhost:8000/embedding"


async def send_request(i):
    async with httpx.AsyncClient() as client:
        texts = [f"This comes from request {i} - sentence {j}" for j in range(50)]
        print(f"Request {i} started")
        start = time.time()
        try:
            response = await client.post(API_URL, json={"inputs": texts}, timeout=60.0)
            duration = time.time() - start
            if response.status_code == 200:
                print(
                    f"Request {i} finished in {duration:.2f}s. Status: {response.status_code}"
                )
                msg = f"Request {i} passed"
            else:
                print(
                    f"Request {i} failed in {duration:.2f}s. Status: {response.status_code}"
                )
                msg = f"Request {i} failed ({response.status_code})"
        except Exception as e:
            print(f"Request {i} Error: {e}")
            msg = f"Request {i} error"
        return msg


async def main():
    print("Starting concurrent stress test...")
    tasks = [send_request(i) for i in range(5)]  # 5 concurrent requests
    results = await asyncio.gather(*tasks)
    print("\nTest Summary:")
    for r in results:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
