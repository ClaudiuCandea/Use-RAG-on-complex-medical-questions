from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import weaviate


def refresh_token():
    path = "C:\\Users\\claud\\Downloads\\sonic-falcon-419513-c01a5a7d9907.json"
    credentials = Credentials.from_service_account_file(
       path,
        scopes=[
            "https://www.googleapis.com/auth/generative-language",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )
    request = Request()
    credentials.refresh(request)
    client = weaviate.connect_to_local(
        headers={"X-PaLM-Api-Key": credentials.token})
    return client


