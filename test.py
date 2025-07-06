from duckduckgo_search import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text("教皇 支持 特朗普", max_results=5))
    print(results)