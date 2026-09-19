import http.server
import os
import sys
import urllib.parse
import webbrowser
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load environment variables from .env in the script directory
load_dotenv(Path(__file__).resolve().parent / ".env")

CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID") or os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    print("[Error] Missing LINKEDIN_CLIENT_ID or LINKEDIN_CLIENT_SECRET in .env file.")
    print("Please set them in your .env file before running this script.")
    sys.exit(1)

REDIRECT_URI = "http://localhost:8000/callback"
SCOPES = "w_member_social openid profile"

code = None

class OAuthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global code
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        
        if "code" in params:
            code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authorization Successful! You can close this tab now.</h1>")
        else:
            self.send_response(400)
            self.end_headers()

# 1. Open Authorization URL in browser
auth_url = (
    f"https://www.linkedin.com/oauth/v2/authorization?"
    f"response_type=code&client_id={CLIENT_ID}&"
    f"redirect_uri={urllib.parse.quote(REDIRECT_URI)}&"
    f"scope={urllib.parse.quote(SCOPES)}"
)

print("Opening browser for LinkedIn authorization...")
webbrowser.open(auth_url)

# 2. Start local server to catch the redirect code
server = http.server.HTTPServer(("localhost", 8000), OAuthHandler)
server.handle_request()

if code:
    # 3. Exchange authorization code for Access Token
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    token_res = requests.post(token_url, data=payload, headers=headers).json()
    access_token = token_res.get("access_token")

    # 4. Fetch your Personal Member URN
    user_info_res = requests.get(
        "https://api.linkedin.com/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"}
    ).json()
    
    person_id = user_info_res.get("sub")
    author_urn = f"urn:li:person:{person_id}"

    print("\n" + "="*50)
    print("SUCCESS! SAVE THESE TO YOUR .env FILE:")
    print("="*50)
    print(f"LINKEDIN_ACCESS_TOKEN={access_token}")
    print(f"LINKEDIN_AUTHOR_URN={author_urn}")
    print("="*50)