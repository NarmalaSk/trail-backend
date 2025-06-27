from flask import Flask, redirect, request, jsonify
import requests
from flask_cors import CORS
import asyncio
import httpx
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from chatbot_response import get_chatbot_response
import os

app = Flask(__name__)
CORS(app)


GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI")
# In-memory store for demo
user_store = {}

# GitHub login redirect
@app.route('/github/login')
def github_login():
    return redirect(
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}&scope=read:user"
    )

@app.route('/chatbot', methods=['POST'])
def chatbot_api():
    data = request.json
    message = data.get('message')

    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = get_chatbot_response(message)

    return jsonify({"response": response}), 200


@app.route('/github/callback')
def github_callback():
    code = request.args.get('code')
    if not code:
        return "Authorization failed: no code", 400

    token_res = requests.post(
        'https://github.com/login/oauth/access_token',
        headers={'Accept': 'application/json'},
        data={
            'client_id': GITHUB_CLIENT_ID,
            'client_secret': GITHUB_CLIENT_SECRET,
            'code': code
        }
    )
    token_json = token_res.json()
    token = token_json.get('access_token')
    if not token:
        return "Failed to get access token", 400

    user_res = requests.get(
        'https://api.github.com/user',
        headers={'Authorization': f'token {token}'}
    )
    user_data = user_res.json()
    username = user_data.get('login')
    if not username:
        return "Failed to get username", 400

    return redirect(f"http://localhost:3000/dashboard?username={username}&platform=GitHub")


@app.route('/linkedin/login')
def linkedin_login():
    return redirect(
        f"https://www.linkedin.com/oauth/v2/authorization"
        f"?response_type=code"
        f"&client_id={LINKEDIN_CLIENT_ID}"
        f"&redirect_uri={LINKEDIN_REDIRECT_URI}"
        f"&scope=openid%20profile"
    )

@app.route('/linkedin/callback')
def linkedin_callback():
    code = request.args.get('code')
    if not code:
        return "Authorization failed: no code", 400

    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "client_id": LINKEDIN_CLIENT_ID,
        "client_secret": LINKEDIN_CLIENT_SECRET
    }
    token_headers = {"Content-Type": "application/x-www-form-urlencoded"}
    token_res = requests.post(token_url, data=token_data, headers=token_headers)
    token_json = token_res.json()
    access_token = token_json.get("access_token")
    if not access_token:
        return "Failed to get access token", 400

    userinfo_url = "https://api.linkedin.com/v2/userinfo"
    userinfo_headers = {"Authorization": f"Bearer {access_token}"}
    userinfo_res = requests.get(userinfo_url, headers=userinfo_headers)
    userinfo_data = userinfo_res.json()

    name = (
        userinfo_data.get("name") or
        userinfo_data.get("preferred_username") or
        userinfo_data.get("sub")
    )
    if not name:
        return "Failed to get user name", 400

    return redirect(f"http://localhost:3000/dashboard?username={name}&platform=LinkedIn")

# Save user via POST request
@app.route('/save-user', methods=['POST'])
def save_user():
    data = request.json
    username = data.get('username')
    platform = data.get('platform')

    if not username:
        return jsonify({"error": "No username provided"}), 400

    user_store[username] = platform
    print(f"✅ User saved: {username} via {platform}")
    return jsonify({"message": f"User {username} saved successfully"}), 200

@app.route('/users', methods=['GET'])
def get_users():
    return jsonify(user_store), 200

# --- GitHub Career Analyzer Inline ---
GITHUB_GRAPHQL_API = "https://api.github.com/graphql"
GITHUB_REST_API = "https://api.github.com"

class GitHubCareerAnalyzer:
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }

    def graphql_query(self, query: str, variables: dict = {}):
        res = requests.post(GITHUB_GRAPHQL_API, json={"query": query, "variables": variables}, headers=self.headers)
        res.raise_for_status()
        result = res.json()
        if "errors" in result:
            raise Exception("GraphQL error: " + str(result["errors"]))
        return result["data"]

    async def fetch_rest(self, url: str, client: httpx.AsyncClient):
        try:
            r = await client.get(url)
            return r.status_code == 200
        except:
            return False

    async def get_file_presence(self, repo_name: str):
        async with httpx.AsyncClient(headers=self.headers) as client:
            urls = {
                "readme": f"{GITHUB_REST_API}/repos/{self.username}/{repo_name}/readme",
                "contributing": f"{GITHUB_REST_API}/repos/{self.username}/{repo_name}/contents/CONTRIBUTING.md",
                "license": f"{GITHUB_REST_API}/repos/{self.username}/{repo_name}/license",
                "docs": f"{GITHUB_REST_API}/repos/{self.username}/{repo_name}/contents/docs"
            }
            results = await asyncio.gather(*(self.fetch_rest(url, client) for url in urls.values()))
            return dict(zip(urls.keys(), results))

    def get_profile_data(self):
        query = """query($login: String!) {
            user(login: $login) {
                name
                login
                bio
                location
                email
                createdAt
                followers { totalCount }
                following { totalCount }
                contributionsCollection {
                  contributionCalendar {
                    totalContributions
                    weeks {
                      contributionDays {
                        date
                        weekday
                        contributionCount
                      }
                    }
                  }
                }
                repositories(first: 100, ownerAffiliations: OWNER, orderBy: {field: STARGAZERS, direction: DESC}) {
                  nodes {
                    name
                    stargazerCount
                    forkCount
                    isArchived
                    hasIssuesEnabled
                    defaultBranchRef {
                      target {
                        ... on Commit {
                          history(first: 100) {
                            totalCount
                          }
                        }
                      }
                    }
                  }
                }
            }
        }"""
        return self.graphql_query(query, {"login": self.username})

    def process_contributions(self, calendar):
        daily = []
        monthly = defaultdict(int)
        today = datetime.now(timezone.utc).date()
        for week in calendar["weeks"]:
            for day in week["contributionDays"]:
                date_str = day["date"]
                count = day["contributionCount"]
                daily.append({"date": date_str, "count": count})
                month_key = date_str[:7]
                monthly[month_key] += count
        return {"daily": daily, "monthly": dict(monthly)}

    async def analyze(self):
        profile_data = self.get_profile_data()
        user = profile_data["user"]
        calendar = user["contributionsCollection"]["contributionCalendar"]
        activity = self.process_contributions(calendar)
        repos = user["repositories"]["nodes"]
        file_tasks = [self.get_file_presence(r["name"]) for r in repos]
        files_data = await asyncio.gather(*file_tasks)
        summary = {
            "profile": {
                "name": user.get("name"),
                "username": user.get("login"),
                "bio": user.get("bio"),
                "location": user.get("location"),
                "created_at": user.get("createdAt"),
                "followers": user["followers"]["totalCount"],
                "following": user["following"]["totalCount"]
            },
            "activity": activity,
            "repos": [{"name": r["name"], **files_data[i]} for i, r in enumerate(repos)]
        }
        return summary

# New route: POST /analyze-github
@app.route('/analyze-github', methods=['POST'])
def analyze_github():
    data = request.json
    username = data.get('username')
    token = data.get('token')

    if not username or not token:
        return jsonify({"error": "username and token required"}), 400

    analyzer = GitHubCareerAnalyzer(username, token)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(analyzer.analyze())
    loop.close()

    print("✅ Analysis Result:", result)  # Log in console
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)
