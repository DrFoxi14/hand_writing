from __future__ import annotations
import argparse
import uvicorn

def serve():
    parser = argparse.ArgumentParser(description="Run the local HIE web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        print("WARNING: HIE may contain personally identifying handwriting data. Binding to a non-loopback interface exposes it to your network.")
    uvicorn.run("hie.api.main:app",host=args.host,port=args.port,reload=False)

if __name__ == "__main__": serve()
