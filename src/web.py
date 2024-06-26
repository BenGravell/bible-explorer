import requests


def fetch_text_from_url(url, **kwargs):
    response = requests.get(url, **kwargs)
    response.raise_for_status()
    return response.text
