import requests, os
from dotenv import load_dotenv
load_dotenv()

def github_search(query: str) -> str:
  headers = {
      "Authorization" : f"token {os.getenv('GITHUB_TOKEN')}",
      'Accept' : "application/vnd.github.v3+json"
  }

  # Search code in the transformers repo
  search_url = 'https://api.github.com/search/code'
  params = {
      "q" : f"{query} repo:huggingface/transformers",
      "per_page" : 3
  }

  response = requests.get(search_url, headers=headers, params=params)

  if response.status_code != 200:
    return f"Github search failed: {response.status_code}"

  results = response.json().get("items", [])

  if not results:
    return "No matching source files found on Github."

  output= []
  for item in results[:2]:
    file_url = item['url']
    file_response = requests.get(file_url, headers=headers)

    if file_response.status_code != 200:
      continue

    file_data = file_response.json()

    import base64
    content = base64.b64decode(file_data["content"]).decode("utf-8", errors="ignore")

    lines = content.split("\n")
    query_term = query.split()[0].lower()  # use first word as anchor

    relevant_lines = []
    for i, line in enumerate(lines):
        if query_term in line.lower():
            # grab surrounding context
            start = max(0, i - 10)
            end = min(len(lines), i + 30)
            relevant_lines.extend(lines[start:end])
            break  # first match is usually enough

    if relevant_lines:
        snippet = "\n".join(relevant_lines[:50])  # cap at 50 lines
    else:
        snippet = "\n".join(lines[:50])  # fallback: first 50 lines

    output.append(f"File: {item['path']}\n\n{snippet}")

  return "\n\n---\n\n".join(output) if output else "Found files but could not extract content."

