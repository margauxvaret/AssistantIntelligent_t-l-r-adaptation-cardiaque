import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from datetime import datetime, timedelta
import gc
import os
import re
import calendar

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Cardio-Réadap Pro",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS GLOBAL — Design médical épuré, sombre et professionnel
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

/* Boutons de navigation sidebar */
.nav-btn {
    display: block;
    width: 100%;
    padding: 12px 16px;
    margin: 4px 0;
    background: transparent;
    border: 1px solid #21262d;
    border-radius: 8px;
    color: #8b949e;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s ease;
    letter-spacing: 0.3px;
}

.nav-btn:hover {
    background: #21262d;
    color: #e6edf3;
    border-color: #30363d;
}

.nav-btn.active {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2137 100%);
    border-color: #3b82f6;
    color: #60a5fa;
    box-shadow: 0 0 12px rgba(59, 130, 246, 0.15);
}

/* Titres */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #e6edf3;
}

/* Cards métriques */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 6px 0;
}

.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #3b82f6;
    font-family: 'DM Serif Display', serif;
}

.metric-label {
    font-size: 12px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

/* Alertes / status */
.status-good { color: #3fb950; }
.status-warn { color: #d29922; }
.status-danger { color: #f85149; }

/* Boîte de chat */
.chat-message-user {
    background: #1a3a5c;
    border: 1px solid #2d6aad;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    text-align: right;
    color: #93c5fd;
}

.chat-message-ai {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e6edf3;
    border-left: 3px solid #3b82f6;
}

/* Calendrier */
.cal-day {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 6px;
    min-height: 70px;
    font-size: 12px;
}

.cal-day-header {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-align: center;
    padding: 8px 0;
}

.cal-event-seance {
    background: #1a3a5c;
    border-left: 3px solid #3b82f6;
    border-radius: 4px;
    padding: 2px 5px;
    margin: 2px 0;
    font-size: 10px;
    color: #93c5fd;
}

.cal-event-consult {
    background: #1f2d1f;
    border-left: 3px solid #3fb950;
    border-radius: 4px;
    padding: 2px 5px;
    margin: 2px 0;
    font-size: 10px;
    color: #7ee787;
}

.cal-event-future {
    background: #2d1f2d;
    border-left: 3px solid #a371f7;
    border-radius: 4px;
    padding: 2px 5px;
    margin: 2px 0;
    font-size: 10px;
    color: #d2a8ff;
}

/* Streamlit overrides */
div[data-testid="stButton"] > button {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    transition: all 0.2s;
}

div[data-testid="stButton"] > button:hover {
    border-color: #3b82f6;
    color: #60a5fa;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a5c, #0d2137);
    border-color: #3b82f6;
    color: #60a5fa;
}

.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stSelectbox > div > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}

div[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
}

.stAlert {
    background: #161b22 !important;
    border-radius: 8px !important;
}

/* Divider */
hr {
    border-color: #21262d !important;
}

/* Plotly background fix */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# BASE DE DONNÉES PATIENTS
# ============================================================================

PATIENTS_DB = {
    "Patient 1 — Post-Infarctus (Stable)": {
        "sexe": "Homme", "age": 58, "poids": 85, "taille": 175,
        "antecedents": "Infarctus du myocarde (il y a 4 mois), stent posé.",
        "medicaments": "Bêta-bloquants (Bisoprolol), Aspirine, Statines.",
        "arrhythmias": "Aucune", "bp_sys": 128, "bp_dia": 82, "rest_hr": 62,
        "niveau_activite": "Sédentaire (reprise)", "max_hr": 162,
        "pathologie": "infarctus_myocarde",
        "phase_rehabilitation": "consolidation",
        "medecin": "Dr. Sophie Renard",
        "specialite": "Cardiologie — Réadaptation"
    },
    "Patient 2 — Hypertension & Arythmie": {
        "sexe": "Femme", "age": 67, "poids": 70, "taille": 162,
        "antecedents": "Fibrillation Atriale (FA) paroxystique, Hypertension.",
        "medicaments": "Anticoagulants (Apixaban), Anti-arythmiques (Flécaïnide).",
        "arrhythmias": "Modérées", "bp_sys": 145, "bp_dia": 90, "rest_hr": 78,
        "niveau_activite": "Actif modéré", "max_hr": 159,
        "pathologie": "hypertension_arythmie",
        "phase_rehabilitation": "entretien",
        "medecin": "Dr. Marc Lefebvre",
        "specialite": "Cardiologie — Arythmologie"
    },
    "Patient 3 — Insuffisance Cardiaque Débutante": {
        "sexe": "Homme", "age": 72, "poids": 92, "taille": 180,
        "antecedents": "Insuffisance cardiaque (Stade II NYHA), Diabète Type 2.",
        "medicaments": "Inhibiteurs ACE, Diurétiques, Metformine.",
        "arrhythmias": "Minimes (Extrasystoles)", "bp_sys": 115, "bp_dia": 75, "rest_hr": 70,
        "niveau_activite": "Sédentaire", "max_hr": 148,
        "pathologie": "insuffisance_cardiaque",
        "phase_rehabilitation": "precoce",
        "medecin": "Dr. Claire Morin",
        "specialite": "Cardiologie — Insuffisance Cardiaque"
    }
}

# ============================================================================
# DONNÉES CALENDRIER FICTIVES PAR PATIENT
# ============================================================================

def get_patient_calendar_data(patient_key):
    today = datetime.now()

    base_data = {
        "Patient 1 — Post-Infarctus (Stable)": {
            "seances_passees": [
                {"date": today - timedelta(days=28), "duree": 25, "fc_moy": 108, "fc_max": 128, "note": "Bonne tolérance"},
                {"date": today - timedelta(days=21), "duree": 28, "fc_moy": 112, "fc_max": 132, "note": "Légère fatigue"},
                {"date": today - timedelta(days=14), "duree": 30, "fc_moy": 115, "fc_max": 135, "note": "Excellent effort"},
                {"date": today - timedelta(days=7), "duree": 30, "fc_moy": 118, "fc_max": 138, "note": "Progression stable"},
                {"date": today - timedelta(days=2), "duree": 30, "fc_moy": 120, "fc_max": 140, "note": "Très bonne séance"},
            ],
            "consultations_passees": [
                {"date": today - timedelta(days=30), "medecin": "Dr. Sophie Renard", "type": "Bilan mensuel", "note": "Évolution favorable"},
            ],
            "seances_futures": [
                {"date": today + timedelta(days=5), "type": "Séance cardio", "duree_prev": 35},
                {"date": today + timedelta(days=12), "type": "Séance cardio", "duree_prev": 35},
                {"date": today + timedelta(days=19), "type": "Séance cardio + renfo", "duree_prev": 40},
            ],
            "prochaine_consultation": {
                "date": today + timedelta(days=15),
                "medecin": "Dr. Sophie Renard",
                "type": "Bilan de suivi mensuel",
                "lieu": "CHU Pitié-Salpêtrière, Paris",
                "doctolib_url": "https://www.doctolib.fr/cardiologue/paris/fabien-vannier-2392b3ef-2ade-4bc6-940b-d8f44faf9c3e?pid=practice-80045&phs=true&page=1&index=1"
            }
        },
        "Patient 2 — Hypertension & Arythmie": {
            "seances_passees": [
                {"date": today - timedelta(days=21), "duree": 20, "fc_moy": 98, "fc_max": 118, "note": "Episode FA bref"},
                {"date": today - timedelta(days=14), "duree": 25, "fc_moy": 102, "fc_max": 122, "note": "Stable"},
                {"date": today - timedelta(days=7), "duree": 25, "fc_moy": 105, "fc_max": 125, "note": "Bonne séance"},
            ],
            "consultations_passees": [
                {"date": today - timedelta(days=45), "medecin": "Dr. Marc Lefebvre", "type": "Suivi arythmie", "note": "Ajustement Flécaïnide"},
                {"date": today - timedelta(days=10), "medecin": "Dr. Marc Lefebvre", "type": "Contrôle tension", "note": "TA stable"},
            ],
            "seances_futures": [
                {"date": today + timedelta(days=3), "type": "Séance douce", "duree_prev": 25},
                {"date": today + timedelta(days=10), "type": "Séance cardio", "duree_prev": 30},
            ],
            "prochaine_consultation": {
                "date": today + timedelta(days=20),
                "medecin": "Dr. Marc Lefebvre",
                "type": "Suivi arythmie + holter",
                "lieu": "Clinique du Cœur, Lyon",
                "doctolib_url": "https://www.doctolib.fr/cardiologue/paris/fabien-vannier-2392b3ef-2ade-4bc6-940b-d8f44faf9c3e?pid=practice-80045&phs=true&page=1&index=1"
            }
        },
        "Patient 3 — Insuffisance Cardiaque Débutante": {
            "seances_passees": [
                {"date": today - timedelta(days=14), "duree": 15, "fc_moy": 88, "fc_max": 105, "note": "Effort adapté"},
                {"date": today - timedelta(days=7), "duree": 18, "fc_moy": 92, "fc_max": 108, "note": "Légère dyspnée"},
            ],
            "consultations_passees": [
                {"date": today - timedelta(days=20), "medecin": "Dr. Claire Morin", "type": "Évaluation initiale", "note": "Programme débuté"},
            ],
            "seances_futures": [
                {"date": today + timedelta(days=4), "type": "Marche thérapeutique", "duree_prev": 20},
                {"date": today + timedelta(days=11), "type": "Séance douce", "duree_prev": 20},
                {"date": today + timedelta(days=18), "type": "Séance cardio légère", "duree_prev": 25},
            ],
            "prochaine_consultation": {
                "date": today + timedelta(days=8),
                "medecin": "Dr. Claire Morin",
                "type": "Contrôle insuffisance cardiaque",
                "lieu": "Hôpital Lariboisière, Paris",
                "doctolib_url": "https://www.doctolib.fr/cardiologue/paris/fabien-vannier-2392b3ef-2ade-4bc6-940b-d8f44faf9c3e?pid=practice-80045&phs=true&page=1&index=1"
            }
        }
    }

    return base_data.get(patient_key, base_data["Patient 1 — Post-Infarctus (Stable)"])

# ============================================================================
# CHARGEMENT DES DOCUMENTS RAG
# ============================================================================

def load_custom_documents():
    documents = []
    doc_folder = '/content/documents_cardio'

    if os.path.exists(doc_folder):
        for filename in os.listdir(doc_folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(doc_folder, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    doc_id = filename.replace('.txt', '')
                    documents.append((content, f"pathologie_{doc_id}"))
                except Exception:
                    pass

    general_docs = [
        ("Zones FC : <50 ans (70-85%), 50-65 ans (65-80%), >65 ans (60-75% FC max).", "general"),
        ("Échelle de Borg : Maintenir l'effort entre 12 et 14.", "general"),
        ("Signes d'arrêt immédiat : Douleur thoracique, vertiges, essoufflement brutal.", "general"),
        ("Échauffement et retour au calme (5-10 min) sont obligatoires.", "general"),
        ("Réadaptation cardiaque post-infarctus : commencer à 60% FCmax, progresser sur 8-12 semaines.", "general"),
        ("Insuffisance cardiaque NYHA II : séances courtes 15-20 min, surveiller la dyspnée.", "general"),
        ("Fibrillation atriale : éviter les efforts brusques, privilégier les activités d'endurance modérée.", "general"),
        ("Hypertension : surveillance tensionnelle avant/après effort, arrêt si >180/110 mmHg.", "general"),
    ]

    documents.extend(general_docs)
    return documents

def create_rag_index(documents):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_texts = [doc[0] for doc in documents]
    doc_metadata = [doc[1] for doc in documents]
    doc_embeddings = embedder.encode(doc_texts)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings))
    return embedder, index, doc_texts, doc_metadata

def search_relevant_docs(query, embedder, index, doc_texts, doc_metadata, patient_pathologie=None, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k * 2)
    relevant_docs = []
    for idx in indices[0]:
        if patient_pathologie and patient_pathologie in doc_metadata[idx]:
            relevant_docs.insert(0, doc_texts[idx])
        elif doc_metadata[idx] == "general":
            relevant_docs.append(doc_texts[idx])
    return relevant_docs[:k]

# ============================================================================
# FONCTIONS CARDIO
# ============================================================================

def get_hr_zones(patient):
    if patient['age'] < 50:
        return int(0.7 * patient['max_hr']), int(0.85 * patient['max_hr'])
    elif patient['age'] < 65:
        return int(0.65 * patient['max_hr']), int(0.8 * patient['max_hr'])
    else:
        return int(0.6 * patient['max_hr']), int(0.75 * patient['max_hr'])

def simulate_realistic_hr(patient, zone_min, zone_max, duration=30):
    t = np.linspace(0, duration, 120)
    warmup_end = duration * 0.15
    effort_end = duration * 0.85
    hr = np.zeros_like(t)

    for i, te in enumerate(t):
        if te < warmup_end:
            progress = te / warmup_end
            hr[i] = patient['rest_hr'] + (zone_min - patient['rest_hr']) * progress
        elif te < effort_end:
            prog = (te - warmup_end) / (effort_end - warmup_end)
            target = zone_min + (zone_max - zone_min) * min(0.7, prog)
            hr[i] = target + np.random.normal(0, 2)
        else:
            rec = (te - effort_end) / (duration - effort_end)
            hr[i] = zone_max - (zone_max - patient['rest_hr']) * rec

    hr += np.random.normal(0, 1.5, len(t))
    hr = np.clip(hr, patient['rest_hr'], patient['max_hr'])
    return t, hr

def create_interactive_chart(t, hr, zone_min, zone_max, patient):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=hr,
        mode='lines',
        name='FC',
        line=dict(color='#3b82f6', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.08)'
    ))

    fig.add_hrect(y0=zone_min, y1=zone_max, fillcolor="rgba(63,185,80,0.06)",
                  line_width=0, annotation_text="Zone cible",
                  annotation_position="top left",
                  annotation_font_color="#3fb950")

    fig.add_hline(y=zone_min, line_dash="dash", line_color="#3fb950", line_width=1,
                  annotation_text=f"Min {zone_min} bpm", annotation_font_color="#3fb950")
    fig.add_hline(y=zone_max, line_dash="dash", line_color="#d29922", line_width=1,
                  annotation_text=f"Max {zone_max} bpm", annotation_font_color="#d29922")

    fig.update_layout(
        title=None,
        xaxis_title="Temps (min)",
        yaxis_title="FC (bpm)",
        hovermode='x unified',
        template='plotly_dark',
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.8)',
        font=dict(family='DM Sans', color='#8b949e', size=11),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor='#21262d', zerolinecolor='#21262d'),
        yaxis=dict(gridcolor='#21262d', zerolinecolor='#21262d'),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )

    return fig

# ============================================================================
# CHARGEMENT IA
# ============================================================================

@st.cache_resource
def load_models():
    try:
        with st.spinner("🔄 Chargement Mistral-7B-Instruct..."):
            custom_docs = load_custom_documents()
            embedder, index, doc_texts, doc_metadata = create_rag_index(custom_docs)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4"
            )

            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )

            torch.cuda.empty_cache()
            gc.collect()

            return embedder, index, doc_texts, doc_metadata, tokenizer, model

    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {str(e)}")
        return None, None, None, None, None, None

# ============================================================================
# GÉNÉRATION ANALYSE SÉANCE
# ============================================================================

def generate_analysis(patient, effort, relevant_docs, tokenizer, model):
    if effort['zone_min'] <= effort['avg'] <= effort['zone_max']:
        adequation = "OUI"
        avis = f"La FC moyenne ({effort['avg']} bpm) est dans la zone cible. Séance adaptée."
    elif effort['avg'] < effort['zone_min']:
        adequation = "NON — en dessous"
        avis = f"FC moyenne ({effort['avg']} bpm) en dessous de la zone cible ({effort['zone_min']} bpm). Intensité insuffisante."
    else:
        adequation = "NON — au-dessus"
        avis = f"FC moyenne ({effort['avg']} bpm) dépasse la zone cible ({effort['zone_max']} bpm). Séance trop intense."

    prompt = f"""<s>[INST] Tu es un médecin expert en réadaptation cardiaque. Réponds en français.

RÈGLES: Utilise UNIQUEMENT les données fournies. Sois concis et pratique. N'invente rien.

PATIENT: {patient['age']} ans, {patient['antecedents']} Traitements: {patient['medicaments']} FC repos: {patient['rest_hr']} bpm

SÉANCE: FC moy={effort['avg']} bpm, FC max={effort['max']} bpm, Zone={effort['zone_min']}-{effort['zone_max']} bpm, Durée={effort['duration']} min

RÉFÉRENCE: {relevant_docs[0][:300] if relevant_docs else "Recommandations standards réadaptation cardiaque"}

Rédige une analyse structurée:

**1. Bilan:** Adéquation zone cible: {adequation}. {avis}

**2. Recommandations pour la prochaine séance** (3 conseils chiffrés):
- Conseil 1:
- Conseil 2:
- Conseil 3:

**3. Exercices de récupération** (2 exercices post-effort):
- Exercice 1:
- Exercice 2:

**4. Objectif chiffré (FC cible) pour la prochaine séance:** [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
    else:
        response = full_response.strip()

    return response

# ============================================================================
# GÉNÉRATION RÉPONSE CHAT
# ============================================================================

def generate_chat_response(question, patient, chat_history, relevant_docs, tokenizer, model, free_mode=False):
    history_text = ""
    for msg in chat_history[-4:]:
        if msg['role'] == 'user':
            history_text += f"Patient: {msg['content']}\n"
        else:
            history_text += f"Médecin: {msg['content']}\n"

    if free_mode:
        context = "Tu réponds à une question générale de santé cardiaque."
        patient_ctx = ""
    else:
        context = "Tu connais le dossier complet du patient."
        patient_ctx = f"""
DOSSIER PATIENT:
- {patient['age']} ans, {patient['sexe']}
- Pathologie: {patient['antecedents']}
- Traitements: {patient['medicaments']}
- FC repos: {patient['rest_hr']} bpm | FC max: {patient['max_hr']} bpm
- Phase réhab: {patient.get('phase_rehabilitation', 'non précisée')}
"""

    prompt = f"""<s>[INST] Tu es un médecin cardiologue spécialisé en réadaptation. {context} Réponds en français, de façon bienveillante et claire.

{patient_ctx}
RÉFÉRENCE MÉDICALE: {relevant_docs[0][:200] if relevant_docs else "Recommandations cardiologiques standards"}

HISTORIQUE RÉCENT:
{history_text}

QUESTION ACTUELLE: {question}

Réponds de façon concise et utile (3-5 phrases max). Si la question dépasse tes données, oriente vers le médecin traitant. [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
    else:
        response = full_response.strip()

    return response

# ============================================================================
# PAGE 1 — TABLEAU DE BORD & SIMULATION
# ============================================================================

def page_dashboard(patient, selected_patient):
    st.markdown(f"## 🫀 Tableau de Bord — {selected_patient.split('—')[0].strip()}")

    bmi = round(patient['poids'] / ((patient['taille'] / 100) ** 2), 1)
    zone_min, zone_max = get_hr_zones(patient)

    # Métriques vitales
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bmi_status = "Normal" if 18.5 < bmi < 25 else "À surveiller"
        bmi_color = "#3fb950" if 18.5 < bmi < 25 else "#d29922"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{bmi_color}">{bmi}</div>
            <div class="metric-label">IMC — {bmi_status}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        bp_color = "#3fb950" if patient['bp_sys'] < 130 else "#d29922" if patient['bp_sys'] < 140 else "#f85149"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{bp_color}">{patient['bp_sys']}<span style="font-size:16px;color:#8b949e">/{patient['bp_dia']}</span></div>
            <div class="metric-label">Tension mmHg</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        hr_color = "#3fb950" if 50 <= patient['rest_hr'] <= 70 else "#d29922"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{hr_color}">{patient['rest_hr']}</div>
            <div class="metric-label">FC Repos (bpm)</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#a371f7">{zone_min}–{zone_max}</div>
            <div class="metric-label">Zone Cible (bpm)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown("### 📋 Dossier Clinique")
        st.markdown(f"""
        <div class="metric-card">
            <div style="margin-bottom:10px">
                <span style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.8px">Patient</span><br>
                <span style="font-size:16px;font-weight:600">{patient['sexe']}, {patient['age']} ans</span>
            </div>
            <div style="margin-bottom:10px">
                <span style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.8px">Antécédents</span><br>
                <span style="font-size:13px">{patient['antecedents']}</span>
            </div>
            <div style="margin-bottom:10px">
                <span style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.8px">Traitements</span><br>
                <span style="font-size:13px">{patient['medicaments']}</span>
            </div>
            <div>
                <span style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.8px">Phase réhabilitation</span><br>
                <span style="font-size:13px;color:#a371f7">{patient.get('phase_rehabilitation','—').capitalize()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🏃 Simulation d'Effort")
        duration = st.slider("Durée de la séance (minutes)", 10, 45, 30, 5)

        if st.button("▶  Démarrer la séance", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_ph = st.empty()
            chart_ph = st.empty()

            t, hr_sim = simulate_realistic_hr(patient, zone_min, zone_max, duration)

            for i in range(1, len(t)):
                fig = create_interactive_chart(t[:i], hr_sim[:i], zone_min, zone_max, patient)
                chart_ph.plotly_chart(fig, use_container_width=True, key=f"live_{i}")
                progress_bar.progress(i / len(t))
                status_ph.markdown(f"<span style='color:#8b949e;font-size:12px'>⏱ {int(i/len(t)*100)}% — FC actuelle : <b style='color:#3b82f6'>{int(hr_sim[i-1])} bpm</b></span>", unsafe_allow_html=True)

                if hr_sim[i - 1] > zone_max * 1.1:
                    st.warning("⚠️ FC trop élevée — réduisez l'intensité")

                time.sleep(0.03)

            avg_hr = int(np.mean(hr_sim[int(len(t) * 0.2):]))
            max_hr_val = int(np.max(hr_sim))

            st.session_state['effort_data'] = {
                'avg': avg_hr, 'max': max_hr_val,
                'duration': duration,
                'zone_min': zone_min, 'zone_max': zone_max,
                'timestamp': datetime.now()
            }

            # Ajouter au calendrier
            new_entry = {
                "date": datetime.now(),
                "duree": duration,
                "fc_moy": avg_hr,
                "fc_max": max_hr_val,
                "note": "Séance simulée"
            }
            if 'new_seances' not in st.session_state:
                st.session_state['new_seances'] = []
            st.session_state['new_seances'].append(new_entry)

            st.success("✅ Séance terminée et enregistrée dans le calendrier !")
            st.balloons()

    with col2:
        st.markdown("### 🤖 Analyse IA de la Séance")

        if st.session_state.get('effort_data'):
            effort = st.session_state['effort_data']
            ts = effort.get('timestamp', datetime.now()).strftime("%d/%m/%Y à %H:%M")

            if effort['zone_min'] <= effort['avg'] <= effort['zone_max']:
                status_icon, status_msg, status_col = "🟢", "Dans la zone cible", "#3fb950"
            elif effort['avg'] < effort['zone_min']:
                status_icon, status_msg, status_col = "🟡", "En dessous de la zone", "#d29922"
            else:
                status_icon, status_msg, status_col = "🟠", "Au-dessus de la zone", "#f85149"

            st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                    <span style="color:#8b949e;font-size:11px">Séance du {ts}</span>
                    <span style="color:{status_col};font-size:12px;font-weight:600">{status_icon} {status_msg}</span>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
                    <div>
                        <div style="color:#3b82f6;font-size:22px;font-weight:600">{effort['avg']}</div>
                        <div style="color:#8b949e;font-size:11px">FC moy (bpm)</div>
                    </div>
                    <div>
                        <div style="color:#f85149;font-size:22px;font-weight:600">{effort['max']}</div>
                        <div style="color:#8b949e;font-size:11px">FC max (bpm)</div>
                    </div>
                    <div>
                        <div style="color:#a371f7;font-size:22px;font-weight:600">{effort['duration']}</div>
                        <div style="color:#8b949e;font-size:11px">Durée (min)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.get('models') is not None:
                if st.button("💬 Générer l'analyse experte", type="secondary", use_container_width=True):
                    embedder, index, doc_texts, doc_metadata, tokenizer, model = st.session_state['models']

                    with st.spinner("🔄 Mistral-7B analyse votre séance..."):
                        query = f"{patient['antecedents']} {patient['age']} ans FC {effort['avg']}"
                        relevant_docs = search_relevant_docs(
                            query, embedder, index, doc_texts, doc_metadata,
                            patient_pathologie=patient.get('pathologie'), k=2
                        )
                        response = generate_analysis(patient, effort, relevant_docs, tokenizer, model)

                        st.session_state['session_history'].append({
                            'date': datetime.now().strftime("%d/%m/%Y %H:%M"),
                            'patient': selected_patient,
                            'effort': effort,
                            'analysis': response
                        })

                    st.markdown("#### 📝 Analyse et Recommandations")
                    st.markdown(f"""
                    <div class="metric-card" style="border-left:3px solid #3b82f6">
                        {response.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📚 Sources médicales utilisées"):
                        for i, doc in enumerate(relevant_docs[:2], 1):
                            st.caption(f"**Document {i}:** {doc[:300]}...")

                    # Programme récupération
                    phase = patient.get('phase_rehabilitation', 'consolidation')
                    with st.expander("🧘 Programme de récupération personnalisé"):
                        if phase == "precoce":
                            st.markdown("**Retour au calme (10-15 min) :**\n1. 🚶 Marche lente 5-7 min\n2. 🫁 Cohérence cardiaque (5s/5s × 5 min)\n3. 🪑 Étirements assis\n\n💧 **Hydratation :** 250-500 ml")
                        elif phase == "consolidation":
                            st.markdown("**Retour au calme (8-10 min) :**\n1. 🚶 Marche lente 5 min\n2. 🫁 Cohérence cardiaque 5 min\n3. 🦵 Étirements debout\n\n💧 **Hydratation :** 500 ml")
                        else:
                            st.markdown("**Retour au calme (5-10 min) :**\n1. 🚶 Marche lente 3-5 min\n2. 🫁 Cohérence cardiaque 5 min\n3. 🧘 Étirements complets")

                    st.download_button(
                        label="📥 Exporter l'analyse",
                        data=f"Analyse cardiaque — {datetime.now().strftime('%d/%m/%Y')}\nPatient: {selected_patient}\n\n{response}",
                        file_name=f"analyse_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                    )
            else:
                st.info("⬅️ Chargez l'IA dans la barre latérale pour générer une analyse")
        else:
            st.markdown("""
            <div class="metric-card" style="text-align:center;padding:40px 20px">
                <div style="font-size:40px;margin-bottom:12px">🏃</div>
                <div style="color:#8b949e;font-size:14px">Lancez une séance d'effort<br>pour obtenir une analyse personnalisée</div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.get('session_history'):
                with st.expander("📜 Dernières analyses"):
                    for s in reversed(st.session_state['session_history'][-3:]):
                        st.markdown(f"**{s['date']}** — {s['patient']}")
                        st.caption(s['analysis'][:200] + "...")
                        st.divider()

# ============================================================================
# PAGE 2 — CALENDRIER & SUIVI
# ============================================================================

def page_calendrier(patient, selected_patient):
    st.markdown(f"## 📅 Calendrier & Suivi — {selected_patient.split('—')[0].strip()}")

    cal_data = get_patient_calendar_data(selected_patient)

    # Fusionner avec les nouvelles séances de session
    all_seances_passees = cal_data['seances_passees'].copy()
    if st.session_state.get('new_seances'):
        all_seances_passees.extend(st.session_state['new_seances'])

    # ---- Prochaine consultation - bandeau ----
    next_consult = cal_data['prochaine_consultation']
    days_until = (next_consult['date'] - datetime.now()).days
    days_label = f"dans {days_until} jour{'s' if days_until > 1 else ''}" if days_until > 0 else "aujourd'hui"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a3a2c,#0d2117);border:1px solid #3fb950;border-radius:12px;padding:20px 24px;margin-bottom:24px">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
            <div>
                <div style="color:#3fb950;font-size:11px;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px">🩺 Prochaine consultation — {days_label}</div>
                <div style="font-size:20px;font-weight:600;color:#e6edf3;font-family:'DM Serif Display',serif">{next_consult['medecin']}</div>
                <div style="color:#7ee787;font-size:13px;margin-top:4px">{next_consult['type']}</div>
                <div style="color:#8b949e;font-size:12px;margin-top:6px">📍 {next_consult['lieu']} &nbsp;·&nbsp; 📆 {next_consult['date'].strftime('%A %d %B %Y').capitalize()}</div>
            </div>
            <div style="text-align:right">
                <a href="{next_consult['doctolib_url']}" target="_blank"
                   style="display:inline-block;background:#3fb950;color:#0d1117;padding:10px 20px;border-radius:8px;font-weight:600;font-size:13px;text-decoration:none;letter-spacing:0.3px">
                   Prendre RDV sur Doctolib →
                </a>
                <div style="color:#8b949e;font-size:11px;margin-top:8px">{patient.get('medecin','—')} · {patient.get('specialite','—')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Statistiques rapides ----
    total_seances = len(all_seances_passees)
    total_minutes = sum(s.get('duree', 0) for s in all_seances_passees)
    avg_fc = int(np.mean([s.get('fc_moy', 0) for s in all_seances_passees])) if all_seances_passees else 0
    nb_futures = len(cal_data['seances_futures'])

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in zip(
        [c1, c2, c3, c4],
        [total_seances, total_minutes, avg_fc, nb_futures],
        ["Séances réalisées", "Minutes d'effort", "FC moy (bpm)", "Séances planifiées"],
        ["#3b82f6", "#a371f7", "#f85149", "#3fb950"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Calendrier visuel (mois courant) ----
    st.markdown("### 📆 Vue Mensuelle")

    today = datetime.now()
    year, month = today.year, today.month
    cal = calendar.monthcalendar(year, month)
    month_name = today.strftime("%B %Y").capitalize()

    # Indexer les événements par jour
    events_by_day = {}
    for s in all_seances_passees:
        if s['date'].month == month and s['date'].year == year:
            d = s['date'].day
            events_by_day.setdefault(d, []).append({"type": "passee", "label": f"🏃 {s.get('duree','?')}min · {s.get('fc_moy','?')}bpm"})

    for s in cal_data['seances_futures']:
        if s['date'].month == month and s['date'].year == year:
            d = s['date'].day
            events_by_day.setdefault(d, []).append({"type": "future", "label": f"📌 {s['type']}"})

    for c in cal_data['consultations_passees']:
        if c['date'].month == month and c['date'].year == year:
            d = c['date'].day
            events_by_day.setdefault(d, []).append({"type": "consult", "label": f"🩺 {c['type']}"})

    next_c = cal_data['prochaine_consultation']
    if next_c['date'].month == month and next_c['date'].year == year:
        d = next_c['date'].day
        events_by_day.setdefault(d, []).append({"type": "consult", "label": f"🩺 {next_c['type']}"})

    st.markdown(f"<div style='color:#8b949e;font-size:13px;margin-bottom:12px;text-transform:capitalize'>{month_name}</div>", unsafe_allow_html=True)

    day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    header_cols = st.columns(7)
    for i, dn in enumerate(day_names):
        with header_cols[i]:
            st.markdown(f"<div class='cal-day-header'>{dn}</div>", unsafe_allow_html=True)

    for week in cal:
        week_cols = st.columns(7)
        for i, day in enumerate(week):
            with week_cols[i]:
                if day == 0:
                    st.markdown("<div style='min-height:70px'></div>", unsafe_allow_html=True)
                else:
                    is_today = (day == today.day)
                    day_style = "border:1px solid #3b82f6;" if is_today else ""
                    day_num_style = "color:#3b82f6;font-weight:700;" if is_today else "color:#8b949e;"

                    evs = events_by_day.get(day, [])
                    ev_html = ""
                    for ev in evs[:2]:
                        css_class = {"passee": "cal-event-seance", "future": "cal-event-future", "consult": "cal-event-consult"}.get(ev["type"], "cal-event-seance")
                        ev_html += f"<div class='{css_class}'>{ev['label']}</div>"
                    if len(evs) > 2:
                        ev_html += f"<div style='font-size:9px;color:#8b949e'>+{len(evs)-2} autres</div>"

                    st.markdown(f"""
                    <div class="cal-day" style="{day_style}">
                        <div style="{day_num_style}font-size:13px;margin-bottom:4px">{day}</div>
                        {ev_html}
                    </div>
                    """, unsafe_allow_html=True)

    # ---- Légende ----
    st.markdown("""
    <div style="display:flex;gap:16px;margin-top:12px;flex-wrap:wrap">
        <span style="font-size:11px;color:#93c5fd">🔵 Séance réalisée</span>
        <span style="font-size:11px;color:#d2a8ff">🟣 Séance planifiée</span>
        <span style="font-size:11px;color:#7ee787">🟢 Consultation</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Historique détaillé ----
    col_hist, col_prog = st.columns([1.2, 1])

    with col_hist:
        st.markdown("### 📊 Historique des Séances")
        if all_seances_passees:
            df = pd.DataFrame([{
                "Date": s['date'].strftime("%d/%m/%Y"),
                "Durée (min)": s.get('duree', '—'),
                "FC moy (bpm)": s.get('fc_moy', '—'),
                "FC max (bpm)": s.get('fc_max', '—'),
                "Note": s.get('note', '—')
            } for s in sorted(all_seances_passees, key=lambda x: x['date'], reverse=True)])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Graphique évolution FC
            if len(all_seances_passees) >= 2:
                dates = [s['date'].strftime("%d/%m") for s in sorted(all_seances_passees, key=lambda x: x['date'])]
                fc_moys = [s.get('fc_moy', 0) for s in sorted(all_seances_passees, key=lambda x: x['date'])]

                fig_prog = go.Figure()
                fig_prog.add_trace(go.Scatter(
                    x=dates, y=fc_moys, mode='lines+markers',
                    name='FC moy', line=dict(color='#3b82f6', width=2),
                    marker=dict(size=7, color='#3b82f6')
                ))
                fig_prog.update_layout(
                    title="Évolution FC moyenne",
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(13,17,23,0.8)',
                    height=220,
                    margin=dict(l=30, r=20, t=40, b=30),
                    font=dict(family='DM Sans', color='#8b949e', size=10),
                    xaxis=dict(gridcolor='#21262d'),
                    yaxis=dict(gridcolor='#21262d')
                )
                st.plotly_chart(fig_prog, use_container_width=True)

    with col_prog:
        st.markdown("### 📅 Prochaines Séances")
        for s in cal_data['seances_futures']:
            days = (s['date'] - datetime.now()).days
            label = f"dans {days}j" if days > 0 else "aujourd'hui"
            st.markdown(f"""
            <div class="metric-card" style="border-left:3px solid #a371f7;margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div style="font-size:13px;font-weight:600;color:#e6edf3">{s['type']}</div>
                        <div style="font-size:11px;color:#8b949e;margin-top:2px">📆 {s['date'].strftime('%A %d/%m').capitalize()} · ⏱ {s.get('duree_prev','?')} min prévues</div>
                    </div>
                    <div style="color:#d2a8ff;font-size:12px;font-weight:600">{label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🩺 Consultations Passées")
        for c in sorted(cal_data['consultations_passees'], key=lambda x: x['date'], reverse=True):
            st.markdown(f"""
            <div class="metric-card" style="border-left:3px solid #3fb950;margin-bottom:8px">
                <div style="font-size:13px;font-weight:600;color:#e6edf3">{c['medecin']}</div>
                <div style="font-size:11px;color:#7ee787;margin-top:2px">{c['type']}</div>
                <div style="font-size:11px;color:#8b949e;margin-top:2px">📆 {c['date'].strftime('%d/%m/%Y')} · {c.get('note','')}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3 — ESPACE DISCUSSION IA
# ============================================================================

def page_chat(patient, selected_patient):
    st.markdown(f"## 💬 Discussion avec l'IA — {selected_patient.split('—')[0].strip()}")

    if st.session_state.get('models') is None:
        st.markdown("""
        <div class="metric-card" style="text-align:center;padding:40px 20px;border-color:#d29922">
            <div style="font-size:36px;margin-bottom:12px">⚠️</div>
            <div style="color:#d29922;font-size:15px;font-weight:600">Modèle IA non chargé</div>
            <div style="color:#8b949e;font-size:13px;margin-top:8px">Utilisez le bouton dans la barre latérale pour charger Mistral-7B</div>
        </div>
        """, unsafe_allow_html=True)
        return

    embedder, index, doc_texts, doc_metadata, tokenizer, model = st.session_state['models']

    # Mode chat
    col_mode, col_info = st.columns([1, 2])
    with col_mode:
        free_mode = st.toggle("🌐 Mode questions libres", value=False,
                              help="Désactivé : l'IA connaît votre dossier. Activé : questions générales de santé.")
    with col_info:
        if free_mode:
            st.markdown("<span style='color:#8b949e;font-size:12px'>Mode libre — questions générales de cardiologie</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#3b82f6;font-size:12px'>Mode contextuel — l'IA connaît votre dossier : {patient['antecedents']}</span>", unsafe_allow_html=True)

    # Exemples de questions
    with st.expander("💡 Suggestions de questions"):
        questions_ex = [
            "Quels exercices puis-je faire à la maison ?",
            "Est-ce que je peux reprendre le vélo ?",
            "Comment savoir si mon effort est trop intense ?",
            "Quels sont les signes d'alerte à surveiller ?",
            "Comment améliorer ma fréquence cardiaque de repos ?",
            "Quelle alimentation est recommandée pour moi ?"
        ]
        for q in questions_ex:
            if st.button(q, key=f"ex_{q}", use_container_width=False):
                st.session_state['pending_question'] = q

    st.divider()

    # Historique chat
    chat_key = f"chat_history_{selected_patient}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    chat_history = st.session_state[chat_key]

    # Message de bienvenue
    if not chat_history:
        st.markdown(f"""
        <div class="chat-message-ai">
            <div style="color:#3b82f6;font-size:11px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Assistant Cardio-Réadap</div>
            Bonjour ! Je suis votre assistant de réadaptation cardiaque. {'Je connais votre dossier médical et peux répondre à vos questions personnalisées.' if not free_mode else 'Posez-moi vos questions générales sur la santé cardiaque.'}
            <br><br>Comment puis-je vous aider ?
        </div>
        """, unsafe_allow_html=True)

    # Afficher historique
    for msg in chat_history:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message-user">
                <div style="font-size:11px;color:#60a5fa;margin-bottom:4px">Vous</div>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message-ai">
                <div style="color:#3b82f6;font-size:11px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px">Assistant</div>
                {msg['content'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

    # Zone de saisie
    user_input = st.text_input(
        "Posez votre question...",
        value=st.session_state.pop('pending_question', ''),
        placeholder="Ex: Quels exercices puis-je faire à la maison ?",
        key="chat_input"
    )

    col_send, col_clear = st.columns([3, 1])
    with col_send:
        send = st.button("Envoyer →", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑 Effacer", use_container_width=True):
            st.session_state[chat_key] = []
            st.rerun()

    if send and user_input.strip():
        chat_history.append({"role": "user", "content": user_input.strip()})

        with st.spinner("🔄 L'IA rédige une réponse..."):
            query = user_input.strip()
            relevant_docs = search_relevant_docs(
                query, embedder, index, doc_texts, doc_metadata,
                patient_pathologie=None if free_mode else patient.get('pathologie'), k=2
            )
            response = generate_chat_response(
                question=user_input.strip(),
                patient=patient,
                chat_history=chat_history,
                relevant_docs=relevant_docs,
                tokenizer=tokenizer,
                model=model,
                free_mode=free_mode
            )

        chat_history.append({"role": "assistant", "content": response})
        st.session_state[chat_key] = chat_history
        st.rerun()

# ============================================================================
# MAIN — NAVIGATION & SIDEBAR
# ============================================================================

def main():
    # Initialisation session state
    for key, default in [
        ('effort_data', None),
        ('models', None),
        ('session_history', []),
        ('page', 'dashboard'),
        ('new_seances', []),
        ('pending_question', '')
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        # Logo + titre
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px 0">
            <div style="font-size:36px">🫀</div>
            <div style="font-family:'DM Serif Display',serif;font-size:18px;color:#e6edf3;margin-top:4px">Cardio-Réadap Pro</div>
            <div style="font-size:11px;color:#8b949e;margin-top:2px;letter-spacing:0.5px">TÉLÉRÉADAPTATION CARDIAQUE</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Sélection patient
        st.markdown("<div style='font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px'>Patient</div>", unsafe_allow_html=True)
        selected_patient = st.selectbox("", list(PATIENTS_DB.keys()), label_visibility="collapsed")
        patient = PATIENTS_DB[selected_patient]

        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px 12px;margin:8px 0">
            <div style="font-size:12px;color:#e6edf3;font-weight:500">{patient['sexe']}, {patient['age']} ans</div>
            <div style="font-size:11px;color:#8b949e;margin-top:2px">{patient.get('phase_rehabilitation','—').capitalize()}</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Navigation
        st.markdown("<div style='font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px'>Navigation</div>", unsafe_allow_html=True)

        pages = [
            ("dashboard", "🫀", "Tableau de Bord", "Bilan & Simulation"),
            ("calendrier", "📅", "Calendrier", "Suivi & Consultations"),
            ("chat", "💬", "Discussion IA", "Questions & Conseils"),
        ]

        for page_id, icon, label, sublabel in pages:
            is_active = st.session_state['page'] == page_id
            active_class = "active" if is_active else ""
            if st.button(
                f"{icon}  {label}\n{sublabel}",
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state['page'] = page_id
                st.rerun()

        st.divider()

        # Chargement IA
        ai_loaded = st.session_state['models'] is not None
        if ai_loaded:
            st.markdown("""
            <div style="background:#1a2d1a;border:1px solid #3fb950;border-radius:8px;padding:10px 12px;text-align:center">
                <div style="color:#3fb950;font-size:12px;font-weight:600">✅ IA Chargée</div>
                <div style="color:#8b949e;font-size:10px;margin-top:2px">Mistral-7B-Instruct</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("🚀 Charger l'IA (Mistral)", use_container_width=True, type="primary"):
                models = load_models()
                if models[0] is not None:
                    st.session_state['models'] = models
                    st.success("✅ Mistral-7B chargé !")
                    st.rerun()

        st.markdown(f"""
        <div style="margin-top:16px;font-size:10px;color:#484f58;text-align:center">
            Cardio-Réadap Pro v2.0<br>
            {datetime.now().strftime('%d/%m/%Y')}
        </div>
        """, unsafe_allow_html=True)

    # Rendu de la page active
    current_page = st.session_state['page']

    if current_page == 'dashboard':
        page_dashboard(patient, selected_patient)
    elif current_page == 'calendrier':
        page_calendrier(patient, selected_patient)
    elif current_page == 'chat':
        page_chat(patient, selected_patient)


if __name__ == "__main__":
    main()