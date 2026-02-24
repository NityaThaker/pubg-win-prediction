from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import load_model, get_model, get_feature_cols, predict_winplace
from app.schemas import PlayerStats, PredictionResponse, ScoutingResponse, HealthResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting PUBG Prediction API...")
    load_model()
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title       = "PUBG Win Placement Prediction API",
    description = "LightGBM model trained on 4.4M PUBG matches. MAE: 0.0513.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── archetype logic ────────────────────────────────────────────────────────
def classify_archetype(s):
    heal_score = s.get("heals", 0) + s.get("boosts", 0)
    aggression = s.get("kills", 0) + (s.get("damageDealt", 0) / 100)
    precision  = s.get("headshotKills", 0) / max(s.get("kills", 1), 1)
    mobility   = s.get("rideDistance", 0) / max(s.get("walkDistance", 0) + 1, 1)
    if aggression > 10 and precision > 0.4: return "Precision Predator", "🎯"
    if aggression > 10:                     return "Aggressive Rusher",   "⚔️"
    if heal_score > 8 and s.get("walkDistance", 0) > 2000: return "Survival Specialist", "🛡️"
    if mobility > 1.5:                      return "Vehicle Raider",      "🚗"
    if s.get("walkDistance", 0) > 3000 and s.get("kills", 0) < 2: return "Ghost Walker", "👻"
    if heal_score > 5 and s.get("kills", 0) < 3: return "Passive Survivor", "🏕️"
    return "Balanced Operator", "⚖️"

INTEL = {
    "Precision Predator":  {
        "strengths":  ["Elite headshot accuracy", "Kill efficiency", "Threat elimination"],
        "weaknesses": ["Over-engagement risk", "Third-party vulnerability"],
        "tip": "Your aim is elite. Push rotations earlier — you win close-range duels."
    },
    "Aggressive Rusher":   {
        "strengths":  ["High kill count", "Pressure generation", "Map control"],
        "weaknesses": ["Low survival rate", "Heals underused", "Late-circle positioning"],
        "tip": "Secure heals between fights — dying early wastes your kill lead."
    },
    "Survival Specialist": {
        "strengths":  ["Resource management", "Late-game presence", "Circle awareness"],
        "weaknesses": ["Low kill contribution", "Passive positioning"],
        "tip": "Add 1-2 proactive kills per match — your positioning already gives the edge."
    },
    "Vehicle Raider":      {
        "strengths":  ["Fast rotations", "Quick loot", "Escape ability"],
        "weaknesses": ["Noise attracts enemies", "Exposed in open terrain"],
        "tip": "Ditch the vehicle one zone early. Arrive quietly."
    },
    "Ghost Walker":        {
        "strengths":  ["Map reading", "Low threat profile", "Consistent late-game"],
        "weaknesses": ["Low kill points", "Passive in final circles"],
        "tip": "Take 1 uncontested kill per zone — it compounds your placement score."
    },
    "Passive Survivor":    {
        "strengths":  ["Consistent placements", "Low-risk decisions", "Good healer"],
        "weaknesses": ["Low damage output", "Struggles in final 5 teams"],
        "tip": "Practice final-circle angles — position beats aim at that stage."
    },
    "Balanced Operator":   {
        "strengths":  ["Versatile", "Adaptable", "Consistent"],
        "weaknesses": ["No dominant strength to exploit"],
        "tip": "Identify your team gap and fill it — fragger or anchor depending on need."
    },
}

def band(p):
    if p >= 0.90: return "Top 10% "
    if p >= 0.75: return "Top 25% "
    if p >= 0.50: return "Top 50% "
    if p >= 0.25: return "Bottom 50% "
    return "Bottom 25% "

# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    m = get_model()
    f = get_feature_cols()
    return HealthResponse(
        status        = "ok" if m is not None else "model_not_loaded",
        model_loaded  = m is not None,
        feature_count = len(f) if f else 0,
        version       = "1.0.0",
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(stats: PlayerStats):
    try:
        if get_model() is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        raw  = stats.model_dump()
        pred = predict_winplace(raw)
        pct  = round(pred * 100, 2)
        return PredictionResponse(
            winPlacePerc        = pred,
            win_probability_pct = pct,
            confidence_band     = band(pred),
            message             = f"Predicted finish: top {100 - pct:.1f}% of players."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/scouting", response_model=ScoutingResponse, tags=["Scouting"])
def scouting(stats: PlayerStats):
    try:
        if get_model() is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        raw            = stats.model_dump()
        pred           = predict_winplace(raw)
        arch, emoji    = classify_archetype(raw)
        intel          = INTEL.get(arch, INTEL["Balanced Operator"])
        total_dist     = raw["walkDistance"] + raw["rideDistance"] + raw["swimDistance"]
        return ScoutingResponse(
            winPlacePerc    = pred,
            archetype       = arch,
            archetype_emoji = emoji,
            strengths       = intel["strengths"],
            weaknesses      = intel["weaknesses"],
            tip             = intel["tip"],
            stats_summary   = {
                "kills":          raw["kills"],
                "damage_dealt":   raw["damageDealt"],
                "total_distance": round(total_dist, 1),
                "heals_boosts":   raw["heals"] + raw["boosts"],
                "headshot_rate":  round(raw["headshotKills"] / max(raw["kills"], 1), 2),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scouting failed: {str(e)}")