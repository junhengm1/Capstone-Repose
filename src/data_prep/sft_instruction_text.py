"""Template pools for build_sft.py.

Each pool contains 20 variants to increase wording diversity in the generated
SFT dataset while keeping the task constraints consistent.
"""

INSTRUCTION_NORMAL_POOL = [
    "Answer the user's bushfire risk question using the retrieved structured data. Use only the retrieved values and do not invent numbers.",
    "Respond to the user's bushfire risk query strictly using the retrieved structured data shown in the input. Do not fabricate any field that is not present.",
    "You are a bushfire risk assistant. Reply to the user using only the retrieved structured data provided below.",
    "Use the retrieved property record to answer the bushfire risk question. Do not add any facts outside the retrieved data.",
    "Provide a property-specific bushfire risk answer grounded only in the retrieved structured data.",
    "Answer the query by summarising the retrieved risk probability, risk level, and visible indicators. Do not guess missing values.",
    "Use the retrieved record as the sole source of truth for this bushfire risk response.",
    "Give a clear answer to the bushfire risk question based only on the retrieved fields in the input.",
    "Act as a bushfire risk assistant and answer using the supplied structured property data only.",
    "Return a factual bushfire risk response that stays within the retrieved values.",
    "Use the provided retrieved data to identify the property's bushfire risk level and probability.",
    "Answer the user's property risk question without introducing unsupported numbers or locations.",
    "Base the response entirely on the retrieved structured property record.",
    "Provide a concise but complete bushfire risk answer using only the given retrieved data.",
    "Use the retrieved risk estimate and visible property indicators to answer the user's question.",
    "Do not infer or invent property details. Answer only from the retrieved structured data.",
    "Give the bushfire risk result for the requested PFI using the retrieved data as evidence.",
    "Respond as a careful risk assistant, relying only on the retrieved risk fields and indicators.",
    "Summarise the property's bushfire risk using the retrieved probability, level, and supporting fields.",
    "Answer the bushfire risk lookup request faithfully from the retrieved structured data.",
]

INSTRUCTION_WHY_POOL = [
    "Explain to the user why the property has the given bushfire risk level, using only the retrieved structured data. Do not invent values.",
    "Provide a brief reasoning that justifies the bushfire risk level, grounded in the retrieved structured data. Do not introduce numbers that are not in the input.",
    "You are a bushfire risk assistant. Explain the assigned risk level for the property using only the retrieved data.",
    "Explain the risk classification by referring only to the retrieved probability and visible indicators.",
    "Describe why the retrieved risk level applies, without adding unsupported causes or external information.",
    "Use the retrieved structured data to give a short explanation of the property's risk level.",
    "Justify the bushfire risk level using the retrieved record as the only evidence.",
    "Explain the classification in plain terms while staying faithful to the retrieved fields.",
    "Give a grounded reason for the risk level based on the retrieved probability and available indicators.",
    "Help the user understand the risk rating without inventing additional drivers.",
    "Provide a data-grounded explanation for the risk level shown in the retrieved record.",
    "Explain how the visible retrieved indicators relate to the final bushfire risk level.",
    "Use only the retrieved property data to explain why this risk band was assigned.",
    "State the risk level and explain the main retrieved factors behind it.",
    "Give a careful explanation of the rating, noting when the calibrated probability is the final source of the level.",
    "Explain the risk result with reference to the retrieved probability, level, and available property features.",
    "Do not speculate about causes. Explain the risk level using only the supplied retrieved data.",
    "Clarify the reasoning behind the retrieved bushfire risk classification.",
    "Provide a short, evidence-based explanation of why the property is in this risk band.",
    "Explain the assigned risk band and keep the explanation anchored to the retrieved values.",
]

INSTRUCTION_SHORT_POOL = [
    "Give the user a concise bushfire risk answer based on the retrieved structured data. Keep the response short.",
    "Reply with a short bushfire risk summary using only the retrieved data. Do not add long explanations.",
    "Provide a brief, direct bushfire risk answer based strictly on the retrieved structured data.",
    "Answer in one or two sentences using only the retrieved risk level and probability.",
    "Give a compact response that states the property's risk level and probability.",
    "Keep the answer brief and grounded in the retrieved structured data.",
    "Provide only the essential bushfire risk result from the retrieved record.",
    "Return a short factual answer without extra explanation or unsupported detail.",
    "State the risk level and probability clearly, using the retrieved data only.",
    "Give a direct short answer to the user's property risk question.",
    "Summarise the risk result briefly and avoid lengthy reasoning.",
    "Use the retrieved data to provide a minimal but complete risk answer.",
    "Keep the response concise: risk level, probability, and no invented values.",
    "Answer the lookup request in a short format based on the retrieved fields.",
    "Provide a quick risk summary using only the retrieved property record.",
    "Do not elaborate unless needed. Give the retrieved risk result directly.",
    "Give the user the short version of the bushfire risk result.",
    "Respond briefly while preserving the retrieved level and probability.",
    "Use a concise style and avoid adding details not present in the input.",
    "Return a short, plain answer grounded in the retrieved structured data.",
]

INSTRUCTION_DETAIL_POOL = [
    "Give the user a detailed bushfire risk explanation that walks through the key drivers, using only the retrieved structured data. Do not invent values.",
    "Provide a thorough breakdown of the bushfire risk for the property, grounded in the retrieved structured data. Do not add fields that are not in the input.",
    "You are a bushfire risk assistant. Produce a detailed answer that covers the main contributing factors visible in the retrieved data.",
    "Explain the risk result in detail using the retrieved probability, level, and visible property indicators.",
    "Walk through the retrieved fields that help contextualise the property's bushfire risk.",
    "Provide a structured explanation of the risk estimate and its visible supporting indicators.",
    "Give a detailed but factual answer that does not go beyond the retrieved property data.",
    "Break down the bushfire risk assessment using only the supplied structured fields.",
    "Explain the retrieved risk result with attention to response access, vegetation, zoning, and fire history when available.",
    "Use the retrieved data to describe the main factors related to the property's risk level.",
    "Provide a detailed property-specific risk explanation without inventing missing drivers.",
    "Give a careful, data-grounded breakdown of the visible risk factors.",
    "Explain how the available retrieved indicators relate to the final risk level.",
    "Produce a detailed answer that remains faithful to the retrieved probability and fields.",
    "Discuss the key retrieved indicators behind the risk result in a clear, grounded way.",
    "Give the user a thorough explanation of the retrieved bushfire risk record.",
    "Describe the risk level, probability, and main visible drivers from the retrieved data.",
    "Provide an evidence-based detailed explanation using only the retrieved structured data.",
    "Walk through the property's retrieved risk information and visible indicators step by step.",
    "Give a comprehensive explanation while avoiding unsupported assumptions or external facts.",
]

INSTRUCTION_NOT_FOUND_POOL = [
    "The retrieved structured data is empty because the property identifier was not found. Politely tell the user and ask for a valid identifier.",
    "When the retrieved structured data is None, do not invent a risk value. Inform the user that the property record was not found and ask for clarification.",
    "If no record exists for the requested property, explain that to the user and request another identifier (PFI, postcode, suburb, or coordinates).",
    "Tell the user that no matching property record was retrieved, and do not provide a made-up risk level.",
    "Respond politely that the requested PFI was not found and ask the user to check the identifier.",
    "When there is no retrieved record, refuse to guess and ask for a valid property reference.",
    "Explain that the lookup returned no property data, so a property-specific risk cannot be provided.",
    "Do not fabricate risk information. State that the property record is missing and request a corrected identifier.",
    "If the PFI is absent from the retrieved data, say so clearly and ask for another lookup key.",
    "Handle the missing record by asking the user to provide a valid PFI or alternative location reference.",
    "Tell the user that the property could not be found in the retrieval results.",
    "Make clear that no risk probability or risk level is available because the record was not retrieved.",
    "Politely ask the user to verify the PFI because the retrieved structured data is empty.",
    "State that the requested property identifier did not match any available record.",
    "When no property data is available, ask for a valid PFI, postcode, suburb, or coordinates.",
    "Avoid giving a risk estimate for a missing record and request a corrected property identifier.",
    "Explain that the lookup failed and that a valid property reference is needed.",
    "Tell the user the property was not found and invite them to provide another identifier.",
    "Respond that the retrieved data is empty for this PFI and do not infer a risk value.",
    "Ask for clarification because the requested property record is not present in the retrieved results.",
]

INSTRUCTION_NO_PFI_POOL = [
    "The user did not provide enough information to look up a property. Ask for a PFI, postcode, suburb, or coordinates and do not invent any risk value.",
    "When the user has not provided any property identifier, do not produce a property-specific risk. Politely request a PFI, postcode, suburb, or coordinates.",
    "Reply to the user by asking for a property identifier (PFI, postcode, suburb, or coordinates). Do not fabricate risk values.",
    "Tell the user that a property-specific risk lookup needs a valid identifier.",
    "Ask the user to provide enough property information before giving a bushfire risk result.",
    "Do not guess which property the user means. Request a PFI or other location reference.",
    "Explain that you need a property identifier to retrieve the structured risk data.",
    "Politely ask for a PFI, postcode, suburb, or coordinates so the property can be looked up.",
    "When the request lacks a lookup key, ask a clarifying question instead of giving a risk estimate.",
    "State that no property-specific risk can be produced without a property reference.",
    "Ask the user to share a valid property identifier before assessing bushfire risk.",
    "Do not provide generic property risk as if it were specific. Request the missing identifier.",
    "Explain that the retrieved data is unavailable because no PFI or location was provided.",
    "Request the information needed to perform a property lookup.",
    "Ask for a PFI or other property reference and avoid inventing a risk probability.",
    "Tell the user that more information is needed to check the bushfire risk.",
    "Ask the user to provide a PFI, postcode, suburb, or coordinates for the lookup.",
    "Respond with a clarification request because the property was not identified.",
    "Do not assume the user's property. Ask for the identifier needed to retrieve data.",
    "Politely request a valid property reference before giving a bushfire risk answer.",
]

INSTRUCTION_ADVICE_POOL = [
    "The user is asking for practical advice about a property. Briefly restate the retrieved risk and give general bushfire preparedness guidance. Do not promise safety or invent details.",
    "Use the retrieved structured data to summarise the risk and then give general preparedness advice. Avoid over-promising safety or making legal/insurance claims.",
    "Combine the retrieved risk values with general bushfire preparedness guidance. Do not invent details that are not in the input and avoid absolute safety claims.",
    "Give practical bushfire preparedness advice while grounding the response in the retrieved risk result.",
    "Restate the retrieved risk level and provide cautious, general preparedness steps.",
    "Use the retrieved property risk information to frame general safety preparation advice.",
    "Offer sensible preparedness guidance without claiming that any action guarantees safety.",
    "Give advice that is general in nature and consistent with the retrieved risk level.",
    "Summarise the retrieved risk and suggest practical preparation measures without legal or insurance claims.",
    "Provide property-aware but non-prescriptive bushfire preparedness advice from the retrieved data.",
    "Answer the advice request by combining the retrieved risk result with general preparedness recommendations.",
    "Keep the advice cautious, practical, and limited to what can be supported by the retrieved data.",
    "Do not invent property details. Give general bushfire preparation guidance tied to the retrieved risk.",
    "Explain what the user should pay attention to based on the retrieved risk level and indicators.",
    "Provide a brief risk summary followed by general steps such as planning, monitoring alerts, and reducing hazards.",
    "Give measured preparedness guidance and avoid saying the property is safe or unsafe in absolute terms.",
    "Use the retrieved risk record to contextualise practical bushfire readiness advice.",
    "Suggest general next steps appropriate for the retrieved risk result.",
    "Provide cautious advice that supports preparedness without exceeding the retrieved evidence.",
    "Frame the advice around the retrieved probability, risk band, and available indicators.",
]

INSTRUCTION_SIMPLE_POOL = [
    "Re-explain the bushfire risk in plain, non-technical language using only the retrieved structured data. Keep the key numbers but drop the jargon.",
    "Explain the bushfire risk in simple terms based on the retrieved data. Keep the risk level and probability but avoid technical wording.",
    "Translate the bushfire risk into plain language for a non-specialist user, while staying faithful to the retrieved structured data.",
    "Give a plain-English explanation of the retrieved risk result without adding new facts.",
    "Explain the risk level and probability simply, using only the retrieved property data.",
    "Use simple wording to help the user understand the bushfire risk result.",
    "Restate the retrieved risk in everyday language and avoid technical jargon.",
    "Make the explanation accessible to a non-specialist while preserving the retrieved values.",
    "Summarise the risk result in simple terms and do not invent details.",
    "Explain what the retrieved risk level means in plain language.",
    "Keep the explanation easy to understand and grounded in the retrieved data.",
    "Convert the retrieved risk result into a simple, user-friendly explanation.",
    "Use clear everyday language to explain the risk probability and level.",
    "Answer as if the user wants the non-technical version of the retrieved record.",
    "Explain the bushfire risk simply while keeping the core numbers accurate.",
    "Avoid jargon and explain the retrieved risk result in a straightforward way.",
    "Give the plain-language version of the property risk assessment.",
    "Help the user understand the risk result without technical terminology.",
    "Use simple words to describe the retrieved risk level, probability, and main visible factors.",
    "Provide an easy-to-read explanation that stays faithful to the retrieved structured data.",
]

NORMAL_QUERY_TEMPLATES = [
    "Can you check the bushfire risk for PFI {pfi}?",
    "Hi, I need the fire risk for PFI {pfi}.",
    "What's the bushfire risk for property {pfi}?",
    "Could you look up PFI {pfi} for me?",
    "I want to know if PFI {pfi} is low, medium, or high risk.",
    "Can you tell me the risk rating for PFI {pfi}?",
    "Please check property {pfi}'s bushfire risk.",
    "Hey, what does the data say for PFI {pfi}?",
    "Can you give me the risk result for PFI {pfi}?",
    "I'm checking PFI {pfi}; what's its bushfire risk?",
    "What risk level has PFI {pfi} been given?",
    "Could you tell me the risk probability for PFI {pfi}?",
    "Need a quick bushfire risk lookup for PFI {pfi}.",
    "What is the current risk estimate for PFI {pfi}?",
    "Can you pull up the risk info for property {pfi}?",
    "I have PFI {pfi}; can you check the bushfire rating?",
    "What's the fire risk classification for PFI {pfi}?",
    "Please show me the bushfire risk summary for PFI {pfi}.",
    "Can you help me check whether PFI {pfi} is risky?",
    "For PFI {pfi}, what bushfire risk should I expect?",
]

WHY_QUERY_TEMPLATES = [
    "Why is this property marked as {risk_level} risk?",
    "Can you explain why PFI {pfi} is {risk_level}?",
    "Why did PFI {pfi} get a {risk_level} rating?",
    "What makes this one {risk_level} risk?",
    "I'm not sure why PFI {pfi} is {risk_level}; can you explain?",
    "Can you walk me through why the risk level is {risk_level}?",
    "What are the main reasons behind this {risk_level} result?",
    "Why is the final label for PFI {pfi} {risk_level}?",
    "What in the data points to {risk_level} risk here?",
    "Can you help me understand this {risk_level} classification?",
    "Why isn't PFI {pfi} in a different risk band?",
    "What factors are making PFI {pfi} come out as {risk_level}?",
    "Could you explain the logic behind the {risk_level} rating?",
    "How did this property end up as {risk_level} risk?",
    "Why should I read PFI {pfi} as {risk_level} risk?",
    "Can you break down why the model says {risk_level}?",
    "What parts of the record matter for the {risk_level} rating?",
    "I see {risk_level} here; what does that come from?",
    "Why does the retrieved result say PFI {pfi} is {risk_level}?",
    "Can you explain the reason for PFI {pfi}'s {risk_level} risk?",
]

SHORT_QUERY_TEMPLATES = [
    "Just the risk level for PFI {pfi}, please.",
    "Short answer for PFI {pfi}?",
    "Only need the level and probability for PFI {pfi}.",
    "Keep it brief: what's PFI {pfi}'s bushfire risk?",
    "PFI {pfi} - quick answer please.",
    "Quick check on PFI {pfi}.",
    "In one sentence, how risky is PFI {pfi}?",
    "Briefly, what risk band is PFI {pfi} in?",
    "Just give me the key result for PFI {pfi}.",
    "What's the short risk summary for PFI {pfi}?",
    "Short version: what's the risk for PFI {pfi}?",
    "I only need PFI {pfi}'s risk level.",
    "Can you give me a quick answer for PFI {pfi}?",
    "Level and probability for PFI {pfi}, no extra detail.",
    "No long explanation, what is PFI {pfi}'s risk?",
    "Quickly summarize PFI {pfi}'s bushfire risk.",
    "What's the headline result for PFI {pfi}?",
    "PFI {pfi}, compact answer please.",
    "Just tell me if PFI {pfi} is low, medium, or high.",
    "What's PFI {pfi}'s risk in plain short form?",
]

DETAIL_QUERY_TEMPLATES = [
    "Can you explain the main drivers for PFI {pfi}?",
    "What are the big factors behind PFI {pfi}'s bushfire risk?",
    "Give me a more detailed explanation for PFI {pfi}.",
    "What contributes most to the risk result for PFI {pfi}?",
    "I want a proper breakdown of PFI {pfi}'s bushfire risk.",
    "Walk me through why PFI {pfi} has this risk.",
    "Can you go into detail on the risk assessment for PFI {pfi}?",
    "What visible factors matter for PFI {pfi}'s risk?",
    "Tell me the full risk story for property {pfi}.",
    "How do the retrieved fields relate to the risk for PFI {pfi}?",
    "Please break down the bushfire risk result for PFI {pfi}.",
    "What should I pay attention to in PFI {pfi}'s risk data?",
    "Can you give me the detailed version for PFI {pfi}?",
    "Could you walk through the data for PFI {pfi}?",
    "Explain the facilities, vegetation, zoning, and fire history for PFI {pfi}.",
    "What does the risk profile look like for PFI {pfi}?",
    "Can you interpret PFI {pfi}'s risk result in detail?",
    "I need more context on why PFI {pfi} got this result.",
    "Can you give a detailed breakdown of PFI {pfi}'s bushfire risk?",
    "Walk through the main risk indicators for PFI {pfi}.",
]

NOT_FOUND_QUERY_TEMPLATES = [
    "Can you check PFI {pfi}?",
    "What's the bushfire risk for PFI {pfi}?",
    "Please look up property {pfi}.",
    "I want to check the fire risk for PFI {pfi}.",
    "Could you search PFI {pfi} for me?",
    "Can you get the risk result for PFI {pfi}?",
    "What risk level does PFI {pfi} have?",
    "Can you find PFI {pfi} in the data?",
    "Need the bushfire risk for PFI {pfi}.",
    "Can you find the property record for PFI {pfi}?",
    "Tell me the fire risk for PFI {pfi}.",
    "Could you try looking up PFI {pfi}?",
    "What is the risk probability for PFI {pfi}?",
    "Please check whether PFI {pfi} is high risk.",
    "Can you give me a risk summary for PFI {pfi}?",
    "Look up the bushfire risk record for PFI {pfi}.",
    "What does the dataset say about PFI {pfi}?",
    "Can you assess this PFI: {pfi}?",
    "Please find the risk classification for PFI {pfi}.",
    "Could you report the bushfire risk for property {pfi}?",
]

NO_PFI_QUERY_TEMPLATES = [
    "Can you check my bushfire risk?",
    "What is my risk?",
    "Is my property safe from bushfires?",
    "Can you look up the fire risk for me?",
    "I want to know my bushfire risk.",
    "Tell me about my fire risk please.",
    "Could you check the bushfire risk for my home?",
    "Am I in a high-risk area?",
    "How risky is my property for bushfire?",
    "Should I be worried about bushfire risk?",
    "Can you assess my property?",
    "What fire risk applies to me?",
    "I need a bushfire risk check.",
    "Can you give me a property risk estimate?",
    "Please tell me whether my home is at risk.",
    "Can you look up my house?",
    "What's the bushfire risk where I live?",
    "Am I low risk or high risk?",
    "Can you help me understand my home's fire risk?",
    "Can you check the risk for my address?",
]

ADVICE_QUERY_TEMPLATES = [
    "What should I do for this property?",
    "What should I pay attention to?",
    "How should I prepare based on this result?",
    "Any precautions I should think about?",
    "Given this risk, what should I do next?",
    "Do you have any preparedness advice for this property?",
    "What actions should I consider for this risk level?",
    "How can I prepare for bushfire risk here?",
    "What are sensible next steps for this property?",
    "What practical advice comes from this risk result?",
    "Should I take any precautions based on this data?",
    "What should the owner keep in mind?",
    "How should someone respond to this bushfire risk level?",
    "What preparation steps would you suggest?",
    "What does this risk mean for day-to-day preparedness?",
    "Can you give me safety prep advice for this property?",
    "What should I do after seeing this risk rating?",
    "How should I plan around this bushfire risk?",
    "What general precautions make sense here?",
    "What can I do to reduce bushfire exposure?",
]

SIMPLE_QUERY_TEMPLATES = [
    "Can you explain that simply?",
    "What does this mean in plain English?",
    "Make it less technical please.",
    "Explain it like I'm not a specialist.",
    "Can you use simpler words?",
    "Plain English version please.",
    "Can you make that easier to understand?",
    "What does this mean in simple terms?",
    "Explain this like I'm new to bushfire risk.",
    "Can you simplify the result?",
    "Give me the non-technical version.",
    "What's the simple takeaway?",
    "Can you explain it without jargon?",
    "Put this in everyday language please.",
    "Make the explanation clearer and simpler.",
    "What should I take away from this?",
    "Can you summarize this simply?",
    "Give me a plain-language explanation.",
    "What does this risk level mean for a normal person?",
    "Can you explain the result more simply?",
]


# ---------------------------------------------------------------------------
# Curtailment / grid constraint focus
# ---------------------------------------------------------------------------

INSTRUCTION_CURTAILMENT_POOL = [
    "Analyse the grid and curtailment risk for this renewable energy site using only the retrieved structured data. Do not invent capacity or dispatch figures.",
    "Assess the power curtailment exposure and grid-export constraints visible in the retrieved project data. Stay within the retrieved values.",
    "You are a renewable energy siting analyst. Explain the curtailment and grid-connection risk for this property using only the retrieved data.",
    "Interpret the retrieved curtailment rate and project capacity data in the context of renewable energy development risk.",
    "Provide a curtailment-focused risk commentary grounded in the retrieved P_project, curtailment, and cur_rate fields.",
    "Use the retrieved project and grid data to assess how grid constraints might affect renewable generation viability at this site.",
    "Explain what the retrieved curtailment figures indicate about grid-export pressure at this property.",
    "Give a technically grounded assessment of grid constraint risk using only the retrieved curtailment and capacity data.",
    "Analyse whether the retrieved curtailment rate represents a material constraint for a renewable energy project at this location.",
    "Use the retrieved data to describe the grid and generation risk relevant to renewable energy siting at this property.",
    "Provide a curtailment risk commentary that stays faithful to the retrieved project capacity and curtailment volume figures.",
    "Interpret the retrieved curtailment rate in the context of AEMO dispatch and grid-export constraints.",
    "Explain the energy storage and curtailment risk for this site using only the retrieved structured data.",
    "Give a data-grounded analysis of grid constraint pressure, drawing only from the retrieved curtailment and capacity fields.",
    "Use the retrieved curtailment and capacity data to assess the operational risk for a renewable energy project at this site.",
    "Assess the significance of the retrieved curtailment rate for a developer evaluating this site for a solar or wind project.",
    "Explain how the retrieved grid-context data (P_project, curtailment, cur_rate) relates to renewable project risk.",
    "Provide an expert commentary on grid constraint risk using only the retrieved project and curtailment fields.",
    "Describe the curtailment exposure at this site and its implications for project revenue, based only on the retrieved data.",
    "Using the retrieved data, evaluate whether grid constraints are a material risk factor for renewable development at this property.",
]

CURTAILMENT_QUERY_TEMPLATES = [
    "What is the curtailment situation at PFI {pfi}?",
    "How constrained is the grid connection for PFI {pfi}?",
    "Can you assess the grid and curtailment risk for PFI {pfi}?",
    "What does the curtailment rate tell me about PFI {pfi}?",
    "How much curtailment risk does PFI {pfi} carry?",
    "Is the grid constraint at PFI {pfi} a concern for a renewable project?",
    "What are the energy curtailment figures for PFI {pfi}?",
    "Can you interpret the curtailment data for PFI {pfi}?",
    "What does the retrieved grid data say about PFI {pfi}?",
    "Is there significant curtailment pressure at PFI {pfi}?",
    "What's the curtailment rate for PFI {pfi} and what does it mean?",
    "Can you explain the project capacity and curtailment context for PFI {pfi}?",
    "How does grid congestion look at PFI {pfi}?",
    "What curtailment risk should I expect at PFI {pfi}?",
    "Is PFI {pfi} heavily curtailed in the retrieved data?",
    "What does the generation and dispatch profile look like for PFI {pfi}?",
    "Can you tell me about the energy storage and curtailment risk at PFI {pfi}?",
    "What is the grid-export pressure at PFI {pfi} based on the retrieved data?",
    "How does the curtailment rate at PFI {pfi} compare to what I'd expect?",
    "Explain the curtailment and capacity figures for PFI {pfi}.",
]


# ---------------------------------------------------------------------------
# Site selection / siting decision
# ---------------------------------------------------------------------------

INSTRUCTION_SITING_POOL = [
    "Provide a renewable energy site-screening recommendation for this property. Consider both bushfire exposure and grid/curtailment constraints from the retrieved data. Do not invent values.",
    "You are a renewable energy siting expert. Give a structured go/no-go screening assessment based on the retrieved dual-risk data.",
    "Using the retrieved data, assess whether this property is a viable candidate for renewable energy development, addressing both fire risk and grid constraints.",
    "Give a site-selection recommendation that integrates the retrieved bushfire risk result with the grid and curtailment context.",
    "Provide a structured screening decision for this site, grounding the recommendation in the retrieved risk level, probability, and available feature data.",
    "Act as a project development analyst. Deliver a site-screening assessment that addresses both the fire exposure and the curtailment risk visible in the retrieved data.",
    "Using only the retrieved structured data, give a balanced siting recommendation that covers bushfire risk and grid constraints.",
    "Provide a developer-facing site-screening summary that clearly states whether the property is a low-concern, watch-list, or high-constraint candidate.",
    "Assess the suitability of this property for renewable energy siting based on the retrieved dual-risk result. Do not add unsupported assumptions.",
    "Give a structured siting recommendation anchored in the retrieved risk level and supporting feature data.",
    "You are a site-selection consultant. Summarise the key risk constraints from the retrieved data and give a clear development recommendation.",
    "Deliver a renewable project siting assessment that explicitly addresses both fire risk and grid export constraints from the retrieved data.",
    "Provide a concise site-screening decision with supporting evidence from the retrieved structured data only.",
    "Using the retrieved data, classify this site as a low-concern, moderate-concern, or high-concern candidate for renewable energy development.",
    "Give a development-feasibility commentary grounded in the retrieved bushfire risk and curtailment data.",
    "Provide a structured siting recommendation that identifies the key risk factors and suggests appropriate next steps for this property.",
    "Act as a risk consultant advising on renewable energy site selection. Use only the retrieved data and do not speculate beyond it.",
    "Summarise the retrieved site-screening result and its implications for a developer considering this property for a wind or solar project.",
    "Give a clear, evidence-based siting recommendation using the retrieved risk classification and supporting field values.",
    "Provide a project-development screening assessment that covers both physical risk and grid risk using only the retrieved data.",
]

SITING_QUERY_TEMPLATES = [
    "Is PFI {pfi} a good candidate for a renewable energy project?",
    "Should I consider developing a renewable energy site at PFI {pfi}?",
    "What's your siting recommendation for PFI {pfi}?",
    "Is PFI {pfi} suitable for a solar or wind project?",
    "Can you give me a site-selection assessment for PFI {pfi}?",
    "Should I progress PFI {pfi} as a development site?",
    "What are the main development constraints at PFI {pfi}?",
    "Is PFI {pfi} worth putting in for further due diligence?",
    "How does PFI {pfi} stack up as a renewable energy candidate?",
    "What would you recommend for PFI {pfi} from a siting perspective?",
    "Can you give me a go/no-go screening result for PFI {pfi}?",
    "Is PFI {pfi} a viable site for a grid-connected renewable project?",
    "What are the key concerns for developing PFI {pfi}?",
    "Should a developer be worried about PFI {pfi}?",
    "Can you summarise the siting risk for PFI {pfi}?",
    "What risk factors does a developer need to consider for PFI {pfi}?",
    "How risky is PFI {pfi} as a renewable project site?",
    "Is there enough risk at PFI {pfi} to reconsider development?",
    "What's the site-screening outcome for PFI {pfi}?",
    "Would you flag PFI {pfi} as a concern for a renewable energy project?",
]


# ---------------------------------------------------------------------------
# Dual-risk (bushfire + curtailment) combined assessment
# ---------------------------------------------------------------------------

INSTRUCTION_DUAL_RISK_POOL = [
    "Provide a structured assessment covering both bushfire risk and grid/curtailment risk for this property. Use only the retrieved data and do not invent values.",
    "You are a dual-risk analyst. Deliver a two-part response: first address the fire exposure, then address the grid and curtailment constraints, using only the retrieved data.",
    "Give a combined bushfire and curtailment risk assessment for this property, grounded in the retrieved structured data.",
    "Analyse the two risk dimensions — bushfire exposure and power curtailment — using only the retrieved feature data and risk classification.",
    "Provide a structured dual-risk commentary that separates fire risk indicators from grid constraint indicators, using only the retrieved data.",
    "Act as a renewable energy risk specialist. Deliver a clear two-section assessment: bushfire risk analysis and grid/curtailment risk analysis.",
    "Using the retrieved data, address both the physical fire risk and the energy dispatch/curtailment risk for this site.",
    "Explain both the bushfire and grid dimensions of the retrieved risk result. Keep each section grounded in the retrieved values.",
    "Give a structured dual-risk answer that addresses fire exposure factors and curtailment pressure as separate but connected risk domains.",
    "Provide a comprehensive site risk assessment that covers fire indicators and grid data in clearly labelled sections. Do not add values beyond the retrieved data.",
    "You are a project risk analyst. Give a two-part response addressing fire risk (using is_prone, veg_area, fire_count, etc.) and grid risk (using P_project, curtailment, cur_rate) from the retrieved data.",
    "Deliver a dual-risk assessment for this property that separates the bushfire exposure analysis from the curtailment and dispatch analysis.",
    "Using only the retrieved structured data, provide a two-part risk briefing: one section on fire and vegetation risk, one section on grid constraint and curtailment risk.",
    "Give a combined risk answer that makes clear which retrieved indicators relate to fire exposure and which relate to grid constraints.",
    "Provide a structured site-risk summary that gives equal weight to bushfire risk and power curtailment risk from the retrieved data.",
    "Act as a dual-risk consultant. Explain the fire risk and the grid risk for this site using only the retrieved feature values.",
    "Deliver a risk briefing in two parts: first summarise the bushfire risk using the retrieved fire indicators, then summarise the curtailment risk using the retrieved grid indicators.",
    "Using the retrieved data, provide a parallel assessment of fire exposure and grid constraint risk for this property.",
    "Give a two-domain risk analysis that addresses bushfire exposure and energy curtailment risk as distinct but complementary concerns.",
    "Provide a structured dual-risk response that stays faithful to the retrieved data and clearly separates fire risk findings from grid risk findings.",
]

DUAL_RISK_QUERY_TEMPLATES = [
    "What are the combined bushfire and curtailment risks for PFI {pfi}?",
    "Can you assess both the fire exposure and the grid constraints at PFI {pfi}?",
    "What does the dual-risk assessment look like for PFI {pfi}?",
    "Can you break down the fire risk and curtailment risk separately for PFI {pfi}?",
    "I need both the bushfire and grid risk picture for PFI {pfi}.",
    "What are the two key risk dimensions for PFI {pfi}?",
    "Can you give me a fire risk and curtailment risk summary for PFI {pfi}?",
    "How does PFI {pfi} look across both risk domains?",
    "What are the bushfire and power curtailment concerns at PFI {pfi}?",
    "Can you split the risk analysis into fire and grid sections for PFI {pfi}?",
    "I want to understand both the fire exposure and the energy dispatch risk at PFI {pfi}.",
    "What does the combined site-screening result tell us about fire and grid risk at PFI {pfi}?",
    "Can you separately address fire risk and curtailment risk for PFI {pfi}?",
    "What are the renewable energy risks from both a fire and a grid perspective for PFI {pfi}?",
    "Can you give me a two-part risk assessment for PFI {pfi}?",
    "What are the physical and grid-related risks at PFI {pfi}?",
    "Break down PFI {pfi}'s risk into its bushfire and curtailment components.",
    "I need the fire risk and the curtailment risk side-by-side for PFI {pfi}.",
    "Can you give a structured dual-risk summary for PFI {pfi}?",
    "What does the data say about fire exposure and grid constraints at PFI {pfi}?",
]

# ---------------------------------------------------------------------------
# Extra natural/conversational variants (appended to extend diversity)
# These are more typical of how real developers and analysts phrase questions.
# ---------------------------------------------------------------------------

NORMAL_QUERY_TEMPLATES += [
    "Before I sign off on the due diligence for PFI {pfi}, what does the fire risk say?",
    "Our team is reviewing PFI {pfi} — what does the bushfire screening show?",
    "I've got a client asking about PFI {pfi}. What can I tell them on fire risk?",
    "Quick question — what does the screening say for PFI {pfi}?",
    "We have PFI {pfi} on our shortlist. What's the fire risk story?",
    "I need PFI {pfi}'s bushfire classification for a project brief.",
    "Can you brief me on PFI {pfi}'s fire exposure?",
    "What does the risk record look like for PFI {pfi}?",
]

WHY_QUERY_TEMPLATES += [
    "I need to justify the {risk_level} rating for PFI {pfi} to our risk committee.",
    "Some indicators for PFI {pfi} look mild but it's still {risk_level} — what's the story?",
    "The developer is questioning why PFI {pfi} came back as {risk_level}. What's the answer?",
    "What's driving the {risk_level} assessment for PFI {pfi}? I want to understand the logic.",
    "Can you explain to a non-technical stakeholder why PFI {pfi} has a {risk_level} rating?",
    "Walk me through what the data actually says about PFI {pfi}'s {risk_level} classification.",
]

DETAIL_QUERY_TEMPLATES += [
    "I need a comprehensive risk briefing on PFI {pfi} for our project file.",
    "Walk me through everything the retrieved data shows for PFI {pfi}.",
    "For our due diligence report, give me the full risk analysis for PFI {pfi}.",
    "What does a detailed read of PFI {pfi}'s risk profile look like?",
]

CURTAILMENT_QUERY_TEMPLATES += [
    "The grid team is flagging issues with PFI {pfi}. What does the curtailment data say?",
    "I'm running IRR sensitivity for PFI {pfi}. What are the grid constraint inputs?",
    "For a financial model, what curtailment situation should I expect at PFI {pfi}?",
    "Our project lender is asking about grid risk at PFI {pfi}. What's the picture?",
    "Is there a dispatch constraint issue at PFI {pfi} based on the retrieved data?",
]

SITING_QUERY_TEMPLATES += [
    "We're doing an early-stage screen of sites including PFI {pfi}. Worth progressing?",
    "Our investment memo needs a site risk section for PFI {pfi}. What's the verdict?",
    "Before we spend on a detailed feasibility study for PFI {pfi}, what does the screening show?",
    "Our land team flagged PFI {pfi} as a potential site. What's your screening assessment?",
    "I'm presenting PFI {pfi} to our development committee. Is it a viable candidate?",
]

DUAL_RISK_QUERY_TEMPLATES += [
    "Give me the full picture on PFI {pfi} — fire and grid together.",
    "I need to understand both risk dimensions for PFI {pfi} before I present to the board.",
    "Can you walk through the fire exposure and the curtailment risk for PFI {pfi}?",
    "What's the complete dual-risk story for PFI {pfi}?",
]

ADVICE_QUERY_TEMPLATES += [
    "Given this result, what would you recommend we do next for this property?",
    "What practical steps should a developer take after seeing this result?",
    "How would you advise a landowner with this risk profile?",
    "What's the most important thing to address based on this risk assessment?",
]

NO_PFI_QUERY_TEMPLATES += [
    "I need to check a site but I don't have the PFI yet. How do I find it?",
    "We're looking at renewable projects in regional Victoria. What information do you need?",
    "I have a property address but no PFI. Can you still help?",
    "Can you check the risk for the block next to the substation near Ballarat?",
    "I want to screen five sites at once. How does that work?",
    "What's the risk for the farm we inspected yesterday near the Latrobe Valley?",
    "We're evaluating a site in Gippsland. What do you need to assess it?",
    "Our consultant gave us a risk score of 0.65. What does that correspond to in your model?",
]

# ---------------------------------------------------------------------------
# Methodology / accuracy inquiry templates
# ---------------------------------------------------------------------------

INSTRUCTION_METHODOLOGY_POOL = [
    "Explain to the user how the risk result was produced, including the data sources and modelling approach. Be transparent about the scope and limitations of the screening tool.",
    "Answer the user's question about the reliability or origin of the risk assessment. Describe the underlying data and machine learning pipeline without fabricating details.",
    "Respond to the user's accuracy or methodology question by describing the data sources, the ML model, and what the result represents.",
    "The user is asking how confident they can be in this result. Explain the data and modelling basis, and be clear that this is a screening tool rather than a definitive assessment.",
    "Address the user's question about data quality or model correctness. Reference the geospatial, fire, vegetation, and grid data sources used.",
    "Explain the provenance of the risk score and risk level, including the machine learning training approach and the types of data used.",
    "Help the user understand where this result comes from. Describe the Victoria-specific data sources and the semi-supervised training process.",
    "The user wants to know the basis for the risk classification. Explain the pipeline from raw geospatial data to the retrieved risk probability and score.",
]

METHODOLOGY_QUERY_TEMPLATES = [
    "How do you know this risk assessment is correct for PFI {pfi}?",
    "Where does this data come from for PFI {pfi}?",
    "How confident are you in the {risk_level} rating for PFI {pfi}?",
    "What is the basis for the risk score assigned to PFI {pfi}?",
    "Can I trust this result for PFI {pfi}? How was it determined?",
    "How was the risk level calculated for PFI {pfi}?",
    "What data went into producing the {risk_level} classification for PFI {pfi}?",
    "I want to understand how reliable this result is for PFI {pfi}.",
    "How do I know the {risk_level} result for PFI {pfi} isn't just a guess?",
    "What datasets and methods are behind the risk score for PFI {pfi}?",
    "Can you explain the methodology behind PFI {pfi}'s risk classification?",
    "Is the {risk_level} risk result for PFI {pfi} based on real data?",
]

# ---------------------------------------------------------------------------
# Closing / farewell templates
# ---------------------------------------------------------------------------

INSTRUCTION_CLOSING_POOL = [
    "The user is ending the conversation. Respond with a brief, friendly closing message.",
    "The user is saying thank you or goodbye. Give a short, natural closing response.",
    "Acknowledge the user's farewell or expression of thanks and wish them well.",
    "The user has finished their queries. Offer a warm closing and invite them to return if needed.",
    "Respond appropriately to the user's thank-you or closing message. Keep it brief and professional.",
    "The user is wrapping up the conversation. Respond in a helpful and friendly way.",
    "Provide a brief and polite response to the user's farewell or thank-you message.",
    "Give a short, friendly closing response to end the conversation naturally.",
]

CLOSING_QUERY_TEMPLATES = [
    "Thank you, that's very helpful.",
    "Thanks for your help!",
    "Great, thanks.",
    "That's all I needed, thank you.",
    "Thank you very much.",
    "Thanks, I have what I need.",
    "Cheers, that's perfect.",
    "Appreciate the help!",
    "Thanks, I'll take it from here.",
    "That answers my question, thanks.",
    "OK, thanks a lot.",
    "Thank you, goodbye.",
    "Thanks, that's all for now.",
    "Great work, thank you.",
]
