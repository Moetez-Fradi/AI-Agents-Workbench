from googlesearch import search

def google_search(query, num=5):
    results = []
    for result in search(query, num_results=num, lang="en"):
        results.append(result)
    return results

print(google_search("latest python release"))