import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
key = os.getenv("TAVILY_API_KEY")
print("key set:", bool(key))
try:
    r = TavilyClient(api_key=key).search("test", max_results=1)
    print("OK, results:", len(r["results"]))
except Exception as e:
    print(type(e).__name__, str(e)[:200])