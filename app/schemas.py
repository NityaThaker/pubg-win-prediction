from pydantic import BaseModel, Field

class PlayerStats(BaseModel):
    # kill stats — with realistic PUBG upper bounds
    kills:           int   = Field(0,      ge=0,  le=35)
    killPlace:       int   = Field(1,      ge=1,  le=100)
    killPoints:      float = Field(0.0,    ge=0,  le=2000)
    killStreaks:      int   = Field(0,      ge=0,  le=20)
    headshotKills:   int   = Field(0,      ge=0,  le=35)
    longestKill:     float = Field(0.0,    ge=0,  le=1500)
    DBNOs:           int   = Field(0,      ge=0,  le=40)
    assists:         int   = Field(0,      ge=0,  le=20)

    # damage
    damageDealt:     float = Field(0.0,    ge=0,  le=5000)

    # distance
    walkDistance:    float = Field(0.0,    ge=0,  le=25000)
    rideDistance:    float = Field(0.0,    ge=0,  le=40000)
    swimDistance:    float = Field(0.0,    ge=0,  le=3000)
    roadKills:       int   = Field(0,      ge=0,  le=20)
    vehicleDestroys: int   = Field(0,      ge=0,  le=20)

    # survival / healing
    heals:           int   = Field(0,      ge=0,  le=40)
    boosts:          int   = Field(0,      ge=0,  le=20)
    revives:         int   = Field(0,      ge=0,  le=10)

    # team
    teamKills:       int   = Field(0,      ge=0,  le=4)

    # loot
    weaponsAcquired: int   = Field(0,      ge=0,  le=30)

    # match context
    matchDuration:   int   = Field(1800,   ge=1,  le=2400)
    maxPlace:        int   = Field(100,    ge=1,  le=100)
    numGroups:       int   = Field(50,     ge=1,  le=100)
    rankPoints:      float = Field(0.0,    ge=0,  le=5000)
    winPoints:       float = Field(0.0,    ge=0,  le=5000)

    class Config:
        json_schema_extra = {
            "example": {
                "kills": 5, "killPlace": 3, "killPoints": 1200.0,
                "killStreaks": 2, "headshotKills": 2, "longestKill": 180.5,
                "DBNOs": 3, "assists": 1, "damageDealt": 450.0,
                "walkDistance": 2500.0, "rideDistance": 800.0, "swimDistance": 0.0,
                "roadKills": 0, "vehicleDestroys": 0, "heals": 3, "boosts": 4,
                "revives": 1, "teamKills": 0, "weaponsAcquired": 6,
                "matchDuration": 1800, "maxPlace": 100, "numGroups": 50,
                "rankPoints": 1500.0, "winPoints": 1200.0
            }
        }

class PredictionResponse(BaseModel):
    winPlacePerc:        float
    win_probability_pct: float
    confidence_band:     str
    message:             str

class ScoutingResponse(BaseModel):
    winPlacePerc:    float
    archetype:       str
    archetype_emoji: str
    strengths:       list[str]
    weaknesses:      list[str]
    tip:             str
    stats_summary:   dict

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    feature_count: int
    version:       str