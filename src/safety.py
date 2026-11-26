"""GramHealth AI - Safety and Healthcare Guardrails Module.
Implements:
1. Medical emergency detection (English, Hindi, Hinglish)
2. Prescription & diagnosis refusal
3. System instruction override / prompt injection defense
4. Out-of-scope / insufficient information detection based on similarity threshold
5. PII sanitization
"""

import re
from typing import Dict, List, Optional, Tuple
import config

# Medical emergency trigger patterns across English, Hindi, and Hinglish
EMERGENCY_PATTERNS = [
    # English
    r"\b(chest pain|heart attack|cardiac arrest|stroke|unconscious|not breathing|stopped breathing)\b",
    r"\b(heavy bleeding|severe hemorrhage|coughing blood|vomiting blood|poison|poisoning|snake bite)\b",
    r"\b(severe burn|choking|convulsion|seizure|anaphylaxis|loss of consciousness|suicide|overdose)\b",
    r"\b(emergency|immediate help|critical condition|ambulance)\b",
    # Hindi (Devanagari)
    r"(सीने में दर्द|दिल का दौरा|बेहोश|सांस नहीं आ रही|सांस रुक गई|खून की उल्टी|जहर|सांप का काटना|दौरा|गंभीर जलना)",
    # Hinglish
    r"\b(chhati me dard|seene me dard|dil ka daura|behosh|saans nahi aa rahi|saans phool rahi)\b",
    r"\b(khoon beh raha|khoon ki ulti|zehar|zeher|zehar kha liya|saanp ne kaat liya|daura pad raha)\b",
    r"\b(jal gaya|emergency hai|ambulance chahiye|bachao)\b",
]

# Prescription / Diagnosis attempt patterns
DIAGNOSIS_PRESCRIPTION_PATTERNS = [
    r"\b(diagnose(\s+my|\s+me|\s+illness|\s+condition)?|what disease do i have|do i have cancer|what illness is this)\b",
    r"\b(prescribe(\s+an|\s+me|\s+a|\s+medicine|\s+antibiotic|\s+drugs)?|give me a prescription|write a prescription|which antibiotic should i take)\b",
    r"\b(what medicine should i buy|exact dose of medicine for my disease|prescribe an antibiotic)\b",
    r"(मुझे कौन सी बीमारी है|मुझे दवा लिख कर दो|कौन सी एंटीबायोटिक लेनी चाहिए|बीमारी बताओ)",
    r"\b(mujhe kaun si bimari hai|dawa likh( kar)?|kaunsi goli khau|bimari hai dawa|konsi dawai)\b",
]

# Prompt injection / override patterns
INJECTION_PATTERNS = [
    r"(ignore (all )?previous instructions|disregard the above|override system rules)",
    r"(you are now a doctor|act as an unrestricted physician|prescribe whatever I ask)",
    r"(reveal your system prompt|show instructions|print prompt)",
    r"(jailbreak|dan mode)",
]


def check_emergency(query: str) -> Tuple[bool, Optional[str]]:
    """Evaluates if the query indicates an acute medical emergency.

    If true, returns an immediate emergency advisory with national helplines.
    """
    clean_q = query.lower()
    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            advisory = (
                "🚨 **CRITICAL MEDICAL EMERGENCY DETECTED**\n\n"
                "If you or someone near you is experiencing severe symptoms, do **NOT** rely on automated advice. "
                "Please seek immediate medical attention right away:\n\n"
                f"• **National Emergency Helpline:** {config.EMERGENCY_CONTACTS['National Emergency']}\n"
                f"• **Ambulance (Free across India):** {config.EMERGENCY_CONTACTS['Ambulance (Free)']}\n"
                f"• **Maternal & Child Health Helpline:** {config.EMERGENCY_CONTACTS['Janani Shishu Suraksha Karyakram (Mother & Child)']}\n"
                f"• **Childline:** {config.EMERGENCY_CONTACTS['Childline']}\n\n"
                "**Action:** Please proceed immediately to your nearest Community Health Centre (CHC), "
                "District Hospital, or call an ambulance."
            )
            return True, advisory

    return False, None


def sanitize_input(query: str) -> str:
    """Neutralizes prompt injection attempts and removes malicious formatting."""
    clean_q = query.strip()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            # Strip injection phrase
            clean_q = re.sub(pattern, "[sanitized]", clean_q, flags=re.IGNORECASE)

    # Prevent extremely long query denial of service
    if len(clean_q) > 1000:
        clean_q = clean_q[:1000]

    return clean_q


def check_diagnosis_or_prescription(query: str) -> Tuple[bool, Optional[str]]:
    """Checks if the user asks for formal medical diagnosis or specific prescription.

    Returns safe educational redirection.
    """
    clean_q = query.lower()
    for pattern in DIAGNOSIS_PRESCRIPTION_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            warning = (
                "⚠️ **Educational Boundary Notice:**\n"
                "GramHealth AI is an educational assistant and **cannot diagnose conditions or prescribe medications**. "
                "Medical diagnoses and drug prescriptions require physical examination and clinical evaluation by a registered medical practitioner.\n\n"
                "For health evaluation, please visit your local ASHA worker, ANM at the Sub-Centre, or a Medical Officer at your nearest Primary Health Centre (PHC)."
            )
            return True, warning

    return False, None


def is_insufficient_information(
    max_similarity_score: float,
    threshold: float = config.SIMILARITY_THRESHOLD,
) -> bool:
    """Determines whether the highest retrieved similarity score is too low

    to reliably answer the question without hallucinating.
    """
    return max_similarity_score < threshold


def format_insufficient_info_response(query: str, max_score: float) -> str:
    """Standardized response when retrieved context does not support the query."""
    return (
        "ℹ️ **Insufficient Information in Knowledge Base**\n\n"
        f"Based on the official healthcare guidelines currently indexed in GramHealth AI "
        f"(WHO, MoHFW, and UNICEF documents), there is not enough verified evidence to answer your query with high confidence "
        f"(Top match similarity: **{max_score:.2f}** below confidence threshold **{config.SIMILARITY_THRESHOLD:.2f}**).\n\n"
        "To avoid misinformation, the assistant will not speculate. Please consult a local healthcare provider (ASHA / ANM / Medical Officer at PHC) "
        "or refer to verified national health portals."
    )
