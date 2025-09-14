import json
import random
import time
import requests

# === Gemini API Call (direct HTTP) ===
def call_gemini_api(prompt, gemini_api_key):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return "⚠️ No recommendation generated."


# === ML Stub (replace with your real ML model) ===
def predict_outcome(live_stats):
    return {
        "win": round(random.uniform(0.2, 0.6), 2),
        "draw": round(random.uniform(0.1, 0.4), 2),
        "lose": round(random.uniform(0.2, 0.6), 2)
    }


# === LLM Recommendation Function ===
def llm_recommendation(live_stats, agent_insights, ml_prediction, gemini_api_key):
    prompt = f"""
    You are a professional football performance analyst.

    🔹 Match context:
    - Minute: {live_stats['minute']}
    - Score: {live_stats['team']} {live_stats['score']} - {live_stats['opponent_score']} {live_stats['opponent']}
    - Possession: {live_stats['possession']}%
    - Shots on target: {live_stats['shots_on_target']}
    - Yellow cards: {live_stats['yellow_cards']}
    - Red cards: {live_stats['red_cards']}
    - Avg player speed: {live_stats['avg_player_speed']} km/h

    🔹 Insights from the last 5 matches (JSON from analysis agent): 
    {json.dumps(agent_insights, indent=2)}

    🔹 ML prediction (probabilities):
    {ml_prediction}

    Task:
    - Provide 2–3 **tactical recommendations** for the coach.
    - Base them BOTH on:
        1. Current real-time stats.
        2. Weaknesses, strong points, and successful/failed tactics observed in the last 5 games.
    - Be concise and actionable (avoid generic advice).
    - Highlight urgent risks (cards, fatigue, momentum).
    - Suggest adjustments (substitutions, formations, pressing, attack/defense balance).
    """

    return call_gemini_api(prompt, gemini_api_key).strip()


# === Main Loop ===
def run_realtime_recommender(team, opponent, agent_json_path, gemini_api_key, duration=3):
    with open(agent_json_path, "r") as f:
        agent_data = json.load(f)

    team_insights = agent_data  # whole JSON (already contains weaknesses, strengths, tactics, feedback)

    print(f"⚽ Starting LLM-based recommender for {team} vs {opponent}...\n")

    for minute in range(5, (duration * 5) + 1, 5):
        # Simulated real-time stats (replace with your live API feed)
        live_stats = {
            "minute": f"{minute}:00",
            "team": team,
            "opponent": opponent,
            "score": random.randint(0, 2),
            "opponent_score": random.randint(0, 2),
            "possession": random.randint(40, 70),
            "shots_on_target": random.randint(0, 6),
            "yellow_cards": random.randint(0, 5),
            "red_cards": random.randint(0, 1),
            "avg_player_speed": round(random.uniform(5.5, 8.5), 2)
        }

        ml_pred = predict_outcome(live_stats)
        recs = llm_recommendation(live_stats, team_insights, ml_pred, gemini_api_key)

        print(f"--- Minute {minute} ---")
        print(recs)
        print()

        time.sleep(2)  # simulate real-time delay
