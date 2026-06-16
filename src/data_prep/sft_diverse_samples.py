"""Hand-crafted diverse Q&A samples for SFT dataset augmentation.

Supplements template-generated data with:
- Natural, conversational question styles (not template-like)
- Expert-quality answers with domain reasoning
- Edge cases: mixed fire/grid signals, missing curtailment data
- Negative cases: no PFI, meta questions, multi-PFI requests

Each entry in HAND_CRAFTED_QA is a dict:
  row         : property record dict, or None for no-data samples
  question    : user question string
  instruction : instruction string for the LLM
  answer      : target answer (hand-crafted, expert-quality)
  type        : SFT sample type label for counting
  reason      : (no-data only) reason string passed to compose_no_data_input
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Realistic property records (five contrasting profiles)
# ---------------------------------------------------------------------------

_HIGH_FIRE_LOW_GRID: Dict[str, Any] = {
    "pfi": 361998, "risk_probability": 0.7411, "risk_level": "High",
    "total_facilities_5km": 0, "closest_facility_distance": 6200.0,
    "is_prone": 1.0, "veg_area": 3.2, "evc_mut_1": 0.0, "evc_bcs_0": 0.0,
    "xgroup_2": 1.0, "xgroup_7": 0.0, "fire_count": 2.0, "yrs_since_last_burn": 28.0,
    "P_project": 22.5, "curtailment": 180.0, "cur_rate": 0.008,
}

_LOW_FIRE_HIGH_GRID: Dict[str, Any] = {
    "pfi": 71657, "risk_probability": 0.1200, "risk_level": "Low",
    "total_facilities_5km": 3, "closest_facility_distance": 820.0,
    "is_prone": 0.0, "veg_area": 0.3, "evc_mut_1": 1.0, "evc_bcs_0": 0.0,
    "xgroup_2": 0.0, "xgroup_7": 0.0, "fire_count": 0.0, "yrs_since_last_burn": 4.0,
    "P_project": 48.5, "curtailment": 9200.0, "cur_rate": 0.184,
}

_MEDIUM_MIXED: Dict[str, Any] = {
    "pfi": 2824823, "risk_probability": 0.4300, "risk_level": "Medium",
    "total_facilities_5km": 2, "closest_facility_distance": 2500.0,
    "is_prone": 1.0, "veg_area": 0.8, "evc_mut_1": 0.0, "evc_bcs_0": 1.0,
    "xgroup_2": 0.0, "xgroup_7": 0.0, "fire_count": 1.0, "yrs_since_last_burn": 12.0,
    "P_project": 35.0, "curtailment": 1400.0, "cur_rate": 0.04,
}

_HIGH_NO_GRID_DATA: Dict[str, Any] = {
    "pfi": 431261150, "risk_probability": 0.8200, "risk_level": "High",
    "total_facilities_5km": 0, "closest_facility_distance": 7800.0,
    "is_prone": 1.0, "veg_area": 5.1, "evc_mut_1": 0.0, "evc_bcs_0": 0.0,
    "xgroup_2": 1.0, "xgroup_7": 0.0, "fire_count": 3.0, "yrs_since_last_burn": 35.0,
    "P_project": None, "curtailment": None, "cur_rate": None,
}

_LOW_CLEAN: Dict[str, Any] = {
    "pfi": 3334104, "risk_probability": 0.0800, "risk_level": "Low",
    "total_facilities_5km": 5, "closest_facility_distance": 450.0,
    "is_prone": 0.0, "veg_area": 0.1, "evc_mut_1": 1.0, "evc_bcs_0": 0.0,
    "xgroup_2": 0.0, "xgroup_7": 0.0, "fire_count": 0.0, "yrs_since_last_burn": 3.0,
    "P_project": 15.0, "curtailment": 120.0, "cur_rate": 0.008,
}

# ---------------------------------------------------------------------------
# Hand-crafted Q&A pairs
# ---------------------------------------------------------------------------

HAND_CRAFTED_QA: List[Dict[str, Any]] = [

    # ---- PFI 361998: High fire, low curtailment ----

    {
        "row": _HIGH_FIRE_LOW_GRID,
        "question": "I'm considering a 20 MW solar project at PFI 361998. What should I know before going further?",
        "instruction": (
            "You are a dual-risk site analyst. Assess the development viability of this property "
            "using only the retrieved data, addressing both fire risk and grid constraints. "
            "Do not invent values not present in the retrieved record."
        ),
        "answer": (
            "PFI 361998 comes back as High risk in the combined site-screening model (retrieved probability "
            "74.11%), and that is a material constraint for any investment memo at this stage.\n\n"
            "The fire exposure picture is quite concerning: the parcel intersects a mapped bushfire-prone "
            "area, native vegetation cover (veg_area: 3.2) indicates substantial available fuel, there are "
            "no fire-response facilities within the 5 km screening radius, and the nearest facility is "
            "6,200 m away — a meaningful emergency-response constraint. The historical fire count of 2 and "
            "28 years since last burn add to the profile. The Lower Slopes or Hills Woodlands macro "
            "vegetation group flag is active, meaning slope and wind exposure can accelerate fire spread.\n\n"
            "On the grid side, the picture is more positive. Retrieved project capacity is 22.50 MW with "
            "a curtailment volume of 180.00 MWh and a curtailment rate of 0.0080% — very low. Grid-export "
            "constraint does not appear to be a material concern at this location.\n\n"
            "For a 20 MW solar project, the bottleneck here is clearly fire risk, not grid constraint. "
            "I would treat this site as requiring a formal bushfire engineering assessment and fire authority "
            "consultation before progressing to investment committee. This is a first-pass screening result "
            "and does not substitute for detailed engineering review."
        ),
        "type": "siting_decision",
    },

    {
        "row": _HIGH_FIRE_LOW_GRID,
        "question": "What's the curtailment situation at PFI 361998? Our grid team is asking.",
        "instruction": (
            "Analyse the grid and curtailment risk for this renewable energy site using only the "
            "retrieved structured data. Do not invent capacity or dispatch figures."
        ),
        "answer": (
            "For PFI 361998, the grid and curtailment picture from the retrieved data is actually quite clean:\n\n"
            "- Installed project capacity (P_project): 22.50 MW\n"
            "- Curtailed energy volume: 180.00 MWh\n"
            "- Curtailment rate: 0.0080% — low (limited current grid-export constraint pressure)\n\n"
            "A curtailment rate of 0.008% is well below the threshold that would concern a project "
            "developer. The retrieved data suggests this connection point is not materially constrained "
            "for dispatch purposes.\n\n"
            "For your grid team: the curtailment numbers are clean. The overall screening classification "
            "for PFI 361998 is High risk (74.11%), but that classification is driven by fire exposure, "
            "not grid performance. Grid constraints are not the concern here."
        ),
        "type": "curtailment_focus",
    },

    {
        "row": _HIGH_FIRE_LOW_GRID,
        "question": "My boss wants to know why PFI 361998 keeps coming back as High risk. Help me explain it.",
        "instruction": (
            "Explain to the user why the property has the given bushfire risk level, using only the "
            "retrieved structured data. Do not invent values."
        ),
        "answer": (
            "PFI 361998 is classified as High risk with a retrieved probability of 74.11%, and the "
            "retrieved data gives several clear reasons why that label is well-supported.\n\n"
            "1. Bushfire-prone area overlay: Yes — intersects a mapped bushfire-prone zone, one of the "
            "strongest prior hazard signals in the model.\n\n"
            "2. Vegetation and fuel load: Native vegetation cover (veg_area: 3.2) indicates substantial "
            "available fuel. The Lower Slopes or Hills Woodlands macro vegetation group flag is active, "
            "meaning slope and wind exposure can amplify fire intensity.\n\n"
            "3. Emergency access: Zero fire-response facilities within 5 km, nearest at 6,200 m. Delayed "
            "response time compounds risk once ignition occurs.\n\n"
            "4. Fire history and fuel accumulation: Historical fire count = 2, and 28 years since last "
            "burn — a long burn interval typically indicates accumulated fuel load.\n\n"
            "On the grid side, the curtailment rate is very low (0.0080%), so that dimension is not "
            "contributing to the High classification. The retrieved indicators are consistently elevated "
            "on the fire-exposure side, and together they push the combined classification firmly into "
            "the High band."
        ),
        "type": "why_explanation",
    },

    # ---- PFI 71657: Low fire, high curtailment ----

    {
        "row": _LOW_FIRE_HIGH_GRID,
        "question": "PFI 71657 shows low fire risk but our financial model looks rough. What's the grid story?",
        "instruction": (
            "Assess the power curtailment exposure and grid-export constraints visible in the retrieved "
            "project data. Stay within the retrieved values."
        ),
        "answer": (
            "For PFI 71657, the grid and curtailment data from the retrieved record explains your "
            "financial model concerns:\n\n"
            "- Installed project capacity (P_project): 48.50 MW\n"
            "- Curtailed energy volume: 9,200.00 MWh\n"
            "- Curtailment rate: 0.1840% — elevated (material grid-export constraint pressure)\n\n"
            "A curtailment rate of 0.184% is well above the 0.10% threshold that I would treat as "
            "material. For a 48.5 MW project, 9,200 MWh of curtailed energy represents a significant "
            "revenue drag and suggests the local connection point is operating under genuine dispatch "
            "constraint. Your IRR sensitivity analysis should stress-test this assumption, because the "
            "retrieved rate is not a minor adjustment.\n\n"
            "To confirm: the fire-risk side of PFI 71657 is very clean (12.00% probability, Low risk, "
            "no prone-area overlay, zero historical fire events, good emergency access). The grid "
            "constraint is the material risk here, not fire exposure."
        ),
        "type": "curtailment_focus",
    },

    {
        "row": _LOW_FIRE_HIGH_GRID,
        "question": "PFI 71657 has low fire risk — can I proceed with the renewable development?",
        "instruction": (
            "Provide a renewable energy site-screening recommendation for this property. Consider both "
            "bushfire exposure and grid/curtailment constraints from the retrieved data. Do not invent values."
        ),
        "answer": (
            "The low fire risk at PFI 71657 is genuine — retrieved data gives it a Low classification "
            "(12.00% probability), with no bushfire-prone area overlay, zero historical fire events, a "
            "recent burn four years ago, and good emergency response access (820 m to nearest facility, "
            "3 facilities within 5 km). The fire screening bar is clear.\n\n"
            "However, the fire result does not give you a green light. The grid and curtailment data is "
            "the constraint:\n\n"
            "  - Project capacity (P_project): 48.50 MW\n"
            "  - Curtailed energy volume: 9,200.00 MWh\n"
            "  - Curtailment rate: 0.1840% — elevated grid-export constraint (material project risk)\n\n"
            "A curtailment rate of 0.184% is materially elevated. At this scale, the revenue impact of "
            "9,200 MWh of curtailed generation should be factored into the investment case before "
            "progressing. Grid constraint risk can affect project revenue certainty and IRR as "
            "significantly as physical risk in some scenarios.\n\n"
            "My assessment: PFI 71657 is a fire-cleared site with a grid constraint problem. I would not "
            "proceed without a grid connection study and a revised financial model that accounts for the "
            "retrieved curtailment rate."
        ),
        "type": "siting_decision",
    },

    {
        "row": _LOW_FIRE_HIGH_GRID,
        "question": "Give me a combined fire and grid risk breakdown for PFI 71657.",
        "instruction": (
            "Provide a structured assessment covering both bushfire risk and grid/curtailment risk for "
            "this property. Use only the retrieved data and do not invent values."
        ),
        "answer": (
            "PFI 71657 is assessed across two risk domains. The combined classification is Low risk at "
            "12.00%, but the two dimensions tell very different stories.\n\n"
            "Bushfire Risk Analysis (Low, 12.00%)\n"
            "The retrieved fire risk profile is clean. No bushfire-prone area overlay, minimal native "
            "vegetation cover (veg_area: 0.3), no historical fire events, last burn four years ago. "
            "Emergency response access is good — 3 facilities within 5 km with the nearest at 820 m. "
            "The mosaic EVC structure flag suggests fuel discontinuity that may slow spread.\n\n"
            "Grid and Curtailment Risk Analysis\n"
            "  - Project capacity (P_project): 48.50 MW\n"
            "  - Curtailed energy: 9,200.00 MWh\n"
            "  - Curtailment rate: 0.1840%\n\n"
            "The curtailment rate exceeds 0.10%, indicating material grid-export constraint pressure. "
            "At this level, the project proforma should include a conservative curtailment assumption and "
            "a grid connection study is recommended before committing to development. The contrast with "
            "the fire risk is striking: this site has essentially no fire-exposure concern but faces "
            "genuine grid constraint risk."
        ),
        "type": "dual_risk",
    },

    # ---- PFI 2824823: Medium mixed ----

    {
        "row": _MEDIUM_MIXED,
        "question": "For our quarterly portfolio review, I need a detailed breakdown of PFI 2824823.",
        "instruction": (
            "Give the user a detailed bushfire risk explanation that walks through the key drivers, "
            "using only the retrieved structured data. Do not invent values."
        ),
        "answer": (
            "For PFI 2824823, the retrieved dual-risk assessment returns a Medium classification "
            "(retrieved probability 43.00%). Detailed breakdown:\n\n"
            "Emergency-response accessibility: 2 fire-response facilities within 5 km; nearest facility "
            "approximately 2,500 m away. A moderate response-access position — neither strong nor "
            "severely constrained.\n\n"
            "Vegetation and fuel context: Bushfire-prone area intersection = Yes; native vegetation "
            "cover (veg_area: 0.8) is in a moderate range; EVC bioregional conservation status "
            "endangered flag is active (evc_bcs_0 = 1), which can indicate complex fuel structure.\n\n"
            "Historical fire activity: Cumulative fire count = 1; years since last burn = 12. A single "
            "fire event and moderate burn interval place this in the low-to-moderate range.\n\n"
            "Grid context: Installed capacity 35.00 MW; curtailed energy 1,400.00 MWh; curtailment rate "
            "0.0400%. Moderate range — worth including in financial modelling but not immediately "
            "disqualifying.\n\n"
            "Overall: a watch-list result with no single overriding constraint. The EVC endangered flag "
            "and prone-area status are the features most worth tracking through further due diligence."
        ),
        "type": "detailed_drivers",
    },

    {
        "row": _MEDIUM_MIXED,
        "question": "In simple language — what does Medium risk actually mean for PFI 2824823?",
        "instruction": (
            "Translate the bushfire risk into plain language for a non-specialist user, while staying "
            "faithful to the retrieved structured data."
        ),
        "answer": (
            "Simply put, PFI 2824823 sits in the middle of the risk range — not a clear pass, not a "
            "clear fail.\n\n"
            "The retrieved probability of about 43% and the Medium label mean: the data picks up some "
            "warning signs (the property is in a bushfire-prone area, there has been one historical fire "
            "nearby, and some vegetation indicators are elevated), but not enough to push it into the "
            "High category.\n\n"
            "For grid constraints, the curtailment rate of about 0.04% is in the moderate zone — "
            "something a developer would include in financial modelling, but not the kind of number that "
            "immediately kills a project.\n\n"
            "The plain takeaway: PFI 2824823 deserves a closer look before committing to development, "
            "but it is not an automatic no."
        ),
        "type": "simple_explanation",
    },

    # ---- PFI 431261150: High fire, no curtailment data ----

    {
        "row": _HIGH_NO_GRID_DATA,
        "question": "What are the curtailment figures for PFI 431261150?",
        "instruction": (
            "Analyse the grid and curtailment risk for this renewable energy site using only the "
            "retrieved structured data. Do not invent capacity or dispatch figures."
        ),
        "answer": (
            "The retrieved record for PFI 431261150 does not include project capacity or curtailment "
            "data. The fields P_project, curtailment, and cur_rate are not present in the current data "
            "extract, so grid constraint risk cannot be assessed from the retrieved record.\n\n"
            "What the retrieved data does show is that the fire risk picture is severe: retrieved "
            "probability 82.00% (High risk). Key fire indicators:\n"
            "  - Bushfire-prone area overlay: Yes\n"
            "  - Native vegetation cover (veg_area): 5.1 — very high fuel exposure\n"
            "  - Historical fire count: 3\n"
            "  - Years since last burn: 35 — strong indication of accumulated fuel load\n"
            "  - Nearest fire-response facility: 7,800 m (response access significantly constrained)\n"
            "  - Fire-response facilities within 5 km: 0\n\n"
            "For a developer: the fire risk alone at this level is a major constraint. I would address "
            "fire exposure first, and treat the missing curtailment data as a separate gap that would "
            "need to be filled through a grid connection enquiry."
        ),
        "type": "curtailment_focus",
    },

    {
        "row": _HIGH_NO_GRID_DATA,
        "question": "Run me through the full risk picture for PFI 431261150.",
        "instruction": (
            "Give the user a detailed bushfire risk explanation that walks through the key drivers, "
            "using only the retrieved structured data. Do not invent values."
        ),
        "answer": (
            "The retrieved assessment for PFI 431261150 is High risk, probability 82.00%. This is one "
            "of the more severe results in the screening model.\n\n"
            "Emergency-response accessibility: Zero fire-response facilities within 5 km; nearest at "
            "7,800 m. This is a significant constraint — delayed access compounds risk substantially.\n\n"
            "Vegetation and fuel context: Intersects a mapped bushfire-prone area. Native vegetation "
            "cover (veg_area: 5.1) is very high, indicating substantial available fuel. The Lower Slopes "
            "or Hills Woodlands macro vegetation group flag is active — topographic and wind exposure "
            "can amplify fire intensity at this location.\n\n"
            "Historical fire activity: Fire count = 3; years since last burn = 35. A 35-year burn "
            "interval is a strong indicator of accumulated fuel load and contributes meaningfully to "
            "the High classification.\n\n"
            "Grid and curtailment context: The retrieved record does not include project capacity or "
            "curtailment data. Grid constraint risk cannot be assessed from the current data extract.\n\n"
            "Overall: the fire risk indicators are consistently and severely elevated across access, "
            "vegetation, prone-area status, and fire history. This site requires detailed bushfire "
            "engineering review before any development commitment."
        ),
        "type": "detailed_drivers",
    },

    # ---- PFI 3334104: Low fire, low curtailment, clean ----

    {
        "row": _LOW_CLEAN,
        "question": "We're screening sites for a wind project. PFI 3334104 looks promising. Am I on the right track?",
        "instruction": (
            "Provide a renewable energy site-screening recommendation for this property. Consider both "
            "bushfire exposure and grid/curtailment constraints from the retrieved data. Do not invent values."
        ),
        "answer": (
            "Based on the retrieved dual-risk assessment, PFI 3334104 is a relatively favourable "
            "candidate for renewable energy siting at this screening stage. The combined classification "
            "is Low risk, retrieved probability 8.00%.\n\n"
            "Fire and physical risk indicators:\n"
            "  - Bushfire-prone area overlay: No\n"
            "  - Native vegetation cover (veg_area): 0.1 — minimal fuel exposure\n"
            "  - Historical fire count: 0\n"
            "  - Years since last burn: 3 (recent burn, reduced fuel load)\n"
            "  - Nearest fire-response facility: 450 m (good emergency response access)\n"
            "  - Fire-response facilities within 5 km: 5\n\n"
            "Grid and curtailment risk indicators:\n"
            "  - Project capacity (P_project): 15.00 MW\n"
            "  - Curtailed energy volume: 120.00 MWh\n"
            "  - Curtailment rate: 0.0080% — low current curtailment pressure\n\n"
            "Your instinct is supported. PFI 3334104 does not raise screening flags on either the fire "
            "or grid dimension. I would progress it to the next stage of due diligence, subject to the "
            "usual site-specific checks: land tenure, planning overlays, wind resource, and network "
            "connection capacity. This is a first-pass screening result only."
        ),
        "type": "siting_decision",
    },

    {
        "row": _LOW_CLEAN,
        "question": "What should I tell my project manager about PFI 3334104?",
        "instruction": (
            "Answer the user's bushfire risk question using the retrieved structured data. "
            "Use only the retrieved values and do not invent numbers."
        ),
        "answer": (
            "PFI 3334104 is classified as Low risk, retrieved probability 8.00%.\n\n"
            "For your project manager, the main points are:\n\n"
            "The fire exposure picture is clean across all retrieved indicators: not in a mapped "
            "bushfire-prone area, minimal vegetation cover (veg_area: 0.1), no historical fire events, "
            "last burn three years ago, and strong emergency response access (5 facilities within 5 km, "
            "nearest at 450 m).\n\n"
            "The grid profile is similarly clean: curtailment rate 0.0080%, project capacity 15.00 MW. "
            "Grid-export constraints are not a material concern at this location.\n\n"
            "The message: PFI 3334104 is not raising flags in the dual-risk screening model on either "
            "dimension. Normal due diligence on land, planning, and resource matters still applies."
        ),
        "type": "normal_query",
    },

    # ---- No-data / negative / edge case samples ----

    {
        "row": None,
        "question": "I'm looking at a site near Ballarat for a solar project. What's the bushfire risk?",
        "instruction": (
            "The user did not provide enough information to look up a property. Ask for a PFI, postcode, "
            "suburb, or coordinates and do not invent any risk value."
        ),
        "answer": (
            "To retrieve a specific bushfire and curtailment risk assessment, I need a property "
            "identifier — ideally the PFI (Property Folio Identifier), or alternatively a postcode, "
            "suburb, or coordinates. A general location like 'near Ballarat' does not have enough "
            "precision to perform a property lookup.\n\n"
            "If you have the PFI for the site, please share it and I can pull up the retrieved risk "
            "record immediately."
        ),
        "type": "no_pfi",
        "reason": "The user described a general location but did not provide a PFI, postcode, suburb, or coordinates.",
    },

    {
        "row": None,
        "question": "Tell me more.",
        "instruction": (
            "When the user has not provided any property identifier, do not produce a property-specific "
            "risk. Politely request a PFI, postcode, suburb, or coordinates."
        ),
        "answer": (
            "I can look into that further, but I need to know which property you are asking about. "
            "Please provide the PFI and your specific question, and I can retrieve the relevant risk "
            "data for that property."
        ),
        "type": "no_pfi",
        "reason": "The user sent a follow-up without identifying a property or providing context for a specific lookup.",
    },

    {
        "row": None,
        "question": "Is Victoria generally a high-risk state for bushfires compared to others?",
        "instruction": (
            "The user did not provide enough information to look up a property. Ask for a valid property "
            "identifier and do not invent any risk value."
        ),
        "answer": (
            "This system is designed to assess specific property risk based on retrieved structured data "
            "for Victoria-registered parcels. I am not set up to give reliable general comparisons "
            "between states or broad regional statistics.\n\n"
            "If you have a specific site in mind, please share the PFI and I can pull up its bushfire "
            "exposure and grid risk record."
        ),
        "type": "no_pfi",
        "reason": "The user asked a general regional question without providing a specific property identifier.",
    },

    {
        "row": None,
        "question": "Can you compare PFI 361998 and PFI 71657 side by side?",
        "instruction": (
            "When the user requests a comparison of multiple PFIs, explain the system's single-lookup "
            "scope and ask for one PFI to start with."
        ),
        "answer": (
            "This system retrieves and assesses one property at a time using the PFI as the lookup key. "
            "I am not able to produce a side-by-side comparison in a single response.\n\n"
            "If you share one PFI first, I can return its full dual-risk assessment; you can then follow "
            "up with the second PFI for a separate result and compare them yourself. Which would you "
            "like to start with — PFI 361998 or PFI 71657?"
        ),
        "type": "no_pfi",
        "reason": "The user requested a comparison of two PFIs, which requires two separate retrievals.",
    },

    {
        "row": None,
        "question": "What is a PFI and how do I use it to look up risk?",
        "instruction": (
            "Reply to the user by explaining what a PFI is and ask for one to proceed. "
            "Do not fabricate risk values."
        ),
        "answer": (
            "PFI stands for Property Folio Identifier — Victoria's unique reference number for each "
            "land parcel. In this system, the PFI is the lookup key used to retrieve the dual-risk "
            "record for a specific property, covering both bushfire exposure and power curtailment risk.\n\n"
            "To check a property's risk, include the PFI in your question — for example: 'What is the "
            "bushfire risk for PFI 361998?' and the system will retrieve the structured risk data for "
            "that parcel. If you do not have the PFI, a postcode, suburb, or coordinates can also help."
        ),
        "type": "no_pfi",
        "reason": "The user asked a meta question about PFIs and the lookup system without providing a property identifier.",
    },

    {
        "row": None,
        "question": "How does the risk model work? I want to understand the methodology.",
        "instruction": (
            "Explain that you need a property identifier to retrieve structured risk data. "
            "You may briefly describe the general framework at a high level."
        ),
        "answer": (
            "At a high level, the model integrates property-level features — including bushfire-prone "
            "area status, vegetation cover, historical fire activity, emergency response access, and "
            "grid curtailment data — to produce a calibrated risk probability and a three-band "
            "classification (Low, Medium, High).\n\n"
            "The system is RAG-style: it retrieves the structured record for a specific PFI from the "
            "database and explains what those retrieved values mean in context. It does not infer "
            "property risk without a retrieved record.\n\n"
            "To see how the indicators apply to a specific site, please provide a PFI and I can walk "
            "through the retrieved data for that property."
        ),
        "type": "no_pfi",
        "reason": "The user asked a methodology question without providing a specific property identifier.",
    },
]
