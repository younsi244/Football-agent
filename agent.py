import os
import sys
import time
import requests
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Politeness settings
REQUEST_DELAY = 0.8
MAX_RETRIES = 2

class FootballAgent:
    def __init__(self, gemini_key: str, serper_key: str):
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not set.")
        if not serper_key:
            raise ValueError("SERPER_API_KEY not set.")
        self.gemini_key = gemini_key
        self.serper_key = serper_key

    # -------------------- TheSportsDB --------------------
    def fetch_matches(self, team_name: str, last_n: int = 5) -> List[str]:
        """Fetch last N matches using TheSportsDB free API."""
        try:
            search_url = "https://www.thesportsdb.com/api/v1/json/3/searchteams.php"
            response = requests.get(search_url, params={"t": team_name}, timeout=10)
            teams_data = response.json()
            if not teams_data.get('teams'):
                print(f"Team '{team_name}' not found!")
                return []

            team = teams_data['teams'][0]
            team_id = team['idTeam']
            team_name_full = team['strTeam']

            matches_url = "https://www.thesportsdb.com/api/v1/json/3/eventslast.php"
            matches_response = requests.get(matches_url, params={"id": team_id}, timeout=10)
            matches_data = matches_response.json()
            if not matches_data.get('results'):
                print(f"No recent matches found for {team_name_full}")
                return []

            matches = matches_data['results'][:last_n]
            match_list = []
            for match in matches:
                date = match.get('dateEvent', 'Unknown date')
                home_team = match.get('strHomeTeam', '')
                away_team = match.get('strAwayTeam', '')
                match_list.append(f"{home_team} vs {away_team} - {date}")

            return match_list
        except Exception as e:
            print(f"Error fetching matches: {e}")
            return []

    # -------------------- Serper --------------------
    def _post_serper(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.serper_key}
        try:
            resp = requests.post(url, headers=headers, json=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[Serper] request error: {e}")
            return None

    def _extract_organic_from_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not data:
            return []
        for key in ("organic", "organic_results", "organic_results_list", "items", "results"):
            block = data.get(key)
            if isinstance(block, list) and block:
                out = []
                for item in block:
                    title = item.get("title") or item.get("name") or item.get("heading") or ""
                    link = item.get("link") or item.get("url") or item.get("displayed_link") or item.get("source")
                    snippet = item.get("snippet") or item.get("description") or item.get("snippet_highlighted") or ""
                    out.append({"title": title, "link": link, "snippet": snippet, "raw": item})
                return out
        return []

    def search_official_site(self, team_name: str) -> Optional[str]:
        """Return the first website link from Serper search results."""
        query = f"official website {team_name}"
        params = {"q": query, "num": 10}
        data = self._post_serper(params)
        results = self._extract_organic_from_response(data)
        if results:
            return results[0]["link"]
        return None

    # -------------------- Page content --------------------
    def extract_page_content(self, url: str) -> str:
        """Extract textual content from a webpage (simple HTML cleaning)."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            content = response.text
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            return content[:5000]
        except Exception as e:
            print(f"Error extracting page content from {url}: {e}")
            return ""

    # -------------------- Gemini LLM --------------------
    def call_gemini_api(self, prompt: str) -> str:
        GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 3000}
        }
        try:
            res = requests.post(GEMINI_API_URL, headers=headers, json=data)
            res.raise_for_status()
            result = res.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return ""

    # -------------------- Analyze --------------------
    def analyze_matches(self, match_list: List[str], team_name: str, website_url: str) -> str:
        if not match_list:
            return f"No match data found for {team_name} from {website_url}."
        prompt = f"""
You are a football performance analyst. Produce a strict JSON report (no extra text) about the recent matches below.

Input:
Team: {team_name}
Source: {website_url}
Matches (only use these matches; do not add older data):
{chr(10).join(match_list)}

TASKS (return only valid JSON):
1) For each match, produce:
   - "match": "HomeTeam vs AwayTeam, YYYY-MM-DD"
   - "competition": short name or null
   - "analysis": {{"weaknesses":["..."], "strengths":["..."], "successful_tactics":["..."], "best_placements":["..."], "overall_feedback":"..."}}
2) Include overall "team_summary" with "avg_insights" and "priority_actions".
3) Return only valid JSON.
"""
        return self.call_gemini_api(prompt)

    # -------------------- Run agent --------------------
    def run(self, team_name: str, last_n: int = 5) -> str:
        match_list = self.fetch_matches(team_name, last_n)
        website_url = self.search_official_site(team_name) or "N/A"
        page_content = self.extract_page_content(website_url) if website_url != "N/A" else ""
        feedback_json = self.analyze_matches(match_list, team_name, website_url)
        return feedback_json


# -------------------- CLI Entry --------------------
if __name__ == "__main__":
    if not GEMINI_API_KEY or not SERPER_API_KEY:
        print("Please set GEMINI_API_KEY and SERPER_API_KEY in your environment or .env")
        sys.exit(1)

    agent = FootballAgent(GEMINI_API_KEY, SERPER_API_KEY)
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Barcelona"
    last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    result = agent.run(team_name, last_n)
    print(result)
