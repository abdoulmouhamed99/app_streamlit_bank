# Application Streamlit : prédiction de la souscription à un dépôt à terme (Bank Marketing)
import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing — Dépôt à terme",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Thème visuel
# Palette : fond #F6F7F9 · surface #FFFFFF · texte #1F2937 · bleu nuit #1E2A5A · indigo #3F51B5
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #F6F7F9; --surface: #FFFFFF; --text: #1F2937; --muted: #6B7280;
  --border: #E5E7EB; --primary: #1E2A5A; --accent: #3F51B5; --accent-soft: #EEF0FA;
}
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; color: var(--text); }
.stApp { background: var(--bg); }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 880px; padding-top: 2.5rem; padding-bottom: 4rem; }

.app-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.75rem; }
.app-logo { width: 52px; height: 52px; border-radius: 14px; background: var(--primary);
  display: flex; align-items: center; justify-content: center; font-size: 1.6rem; flex-shrink: 0; }
.app-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; color: var(--primary); line-height: 1.2; }
.app-subtitle { margin: .2rem 0 0; color: var(--muted); font-size: .95rem; }

.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 1rem; font-weight: 500; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--primary); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }

[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface); border: 1px solid var(--border) !important;
  border-radius: 14px; box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
[data-testid="stForm"] { padding: 1.5rem; }

.section-title { font-size: .95rem; font-weight: 600; color: var(--primary); margin: 1.1rem 0 .15rem; }
.section-title:first-child { margin-top: 0; }
.section-hint { font-size: .85rem; color: var(--muted); margin-bottom: .6rem; }

.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px; font-weight: 600; padding: .6rem 1.4rem; transition: background .15s ease;
}
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: var(--primary); border: 1px solid var(--primary); color: #fff; }
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover,
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  background: var(--accent); border-color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

.result-card { background: var(--primary); color: #fff; border-radius: 16px; padding: 1.75rem 2rem; margin: 1.25rem 0 .75rem; }
.result-card.neg { background: #334155; }
.result-label { font-size: .9rem; color: #C7CDEB; margin: 0; }
.result-value { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; margin: .15rem 0 .35rem; line-height: 1.15; }
.result-note { font-size: .88rem; color: #C7CDEB; margin: 0; }

.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin: .5rem 0 1rem; }
.chip { background: var(--accent-soft); color: var(--primary); border-radius: 999px; padding: .3rem .8rem; font-size: .82rem; font-weight: 500; }

[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .9rem 1.1rem; }
[data-testid="stMetricLabel"] { color: var(--muted); }
.stProgress > div > div > div > div { background-color: var(--accent); }

@media (max-width: 640px) {
  .block-container { padding-top: 1.25rem; }
  .app-title { font-size: 1.3rem; }
  .result-value { font-size: 1.7rem; }
  [data-testid="stForm"] { padding: 1rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Chargement des objets sauvegardés (une seule fois grâce au cache)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_objects():
    encoders = joblib.load('encoders.joblib1')   # encodeurs
    xgb_model = joblib.load('xgb.joblib')        # meilleur modèle
    uniques = joblib.load('uniques.joblib1')     # valeurs uniques
    scaler = joblib.load('scaler.joblib1')       # normalisateur
    return encoders, xgb_model, uniques, scaler


try:
    encoders, xgb_model, uniques, scaler = load_objects()
except FileNotFoundError as e:
    st.error(
        f"Fichier introuvable : **{e.filename}**. Placez `encoders.joblib1`, `xgb.joblib`, "
        "`uniques.joblib1` et `scaler.joblib1` dans le même dossier que ce script."
    )
    st.stop()

COLUMNS = ["age", "job", "marital", "education", "housing", "loan", "contact", "month",
           "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome"]
POSITIVE = {"yes", "oui", "1", "true"}


# ----------------------------------------------------------------------------
# Logique de prédiction (identique à la version Gradio)
# ----------------------------------------------------------------------------
def _predict_core(age, job, marital, education, housing, loan,
                  contact, month, day_of_week, duration,
                  campaign, pdays, previous, poutcome):
    # Encoder les variables catégorielles
    job = encoders['job'].transform([job])[0]
    marital = encoders['marital'].transform([marital])[0]
    education = encoders['education'].transform([education])[0]
    housing = encoders['housing'].transform([housing])[0]
    loan = encoders['loan'].transform([loan])[0]
    contact = encoders['contact'].transform([contact])[0]
    month = encoders['month'].transform([month])[0]
    day_of_week = encoders['day_of_week'].transform([day_of_week])[0]
    poutcome = encoders['poutcome'].transform([poutcome])[0]

    # Construire les 14 variables
    x_new = np.array([
        age, job, marital, education, housing, loan,
        contact, month, day_of_week, duration,
        campaign, pdays, previous, poutcome
    ], dtype=float).reshape(1, -1)

    # Normaliser puis prédire
    x_new = scaler.transform(x_new)
    y_pred = xgb_model.predict(x_new)

    # Décoder la prédiction
    label = str(encoders['y'].inverse_transform(np.asarray(y_pred).astype(int))[0])

    # Probabilités par classe (si le modèle les fournit)
    proba = None
    if hasattr(xgb_model, "predict_proba"):
        p = xgb_model.predict_proba(x_new)[0]
        proba = {str(c): float(v) for c, v in zip(encoders['y'].classes_, p)}
    return label, proba


def Pred_func(*args):
    """Retourne uniquement l'étiquette prédite."""
    return _predict_core(*args)[0]


def Pred_func_csv(df):
    """Prédit chaque ligne ; une ligne invalide reçoit None au lieu de bloquer tout le fichier."""
    predictions = []
    for row in df.iloc[:, :14].values:
        try:
            predictions.append(Pred_func(*row))
        except Exception:
            predictions.append(None)
    df = df.copy()
    df['etat'] = predictions
    return df


# ----------------------------------------------------------------------------
# En-tête
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
      <div class="app-logo">🏦</div>
      <div>
        <h1 class="app-title">Prédire la souscription à un dépôt à terme</h1>
        <p class="app-subtitle">Estimez si un client est susceptible de souscrire, à partir de son
        profil et de l'historique des campagnes de marketing.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Prédiction individuelle", "Prédiction multiple", "À propos du modèle"])

# ----------------------------------------------------------------------------
# Onglet 1 : prédiction individuelle
# ----------------------------------------------------------------------------
with tab1:
    with st.form("form_simple"):
        st.markdown('<div class="section-title">Profil du client</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("Âge", min_value=18, max_value=100, value=40, step=1)
        job = c2.selectbox("Profession", options=uniques[0])
        marital = c1.selectbox("Situation familiale", options=uniques[1])
        education = c2.selectbox("Niveau d'études", options=uniques[2])

        st.markdown('<div class="section-title">Situation financière</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        housing = c1.selectbox("Prêt immobilier", options=uniques[3])
        loan = c2.selectbox("Prêt personnel", options=uniques[4])

        st.markdown('<div class="section-title">Dernier contact</div>'
                    '<div class="section-hint">La durée de l\'appel n\'est connue qu\'après l\'appel.</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        contact = c1.selectbox("Mode de contact", options=uniques[5])
        month = c2.selectbox("Mois", options=uniques[6])
        day_of_week = c3.selectbox("Jour de la semaine", options=uniques[7])
        duration = st.number_input("Durée de l'appel (secondes)", min_value=0, value=180, step=10)

        st.markdown('<div class="section-title">Historique des campagnes</div>'
                    '<div class="section-hint">Pour « jours depuis le dernier contact », 999 signifie '
                    'jamais contacté (convention Bank Marketing).</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        campaign = c1.number_input("Contacts (cette campagne)", min_value=0, value=1, step=1)
        previous = c2.number_input("Contacts (avant)", min_value=0, value=0, step=1)
        pdays = c3.number_input("Jours depuis le dernier contact", min_value=0, value=999, step=1)
        poutcome = st.selectbox("Résultat de la campagne précédente", options=uniques[8])

        submit = st.form_submit_button("Prédire", type="primary", use_container_width=True)

    if submit:
        try:
            with st.spinner("Calcul de la prédiction…"):
                label, proba = _predict_core(age, job, marital, education, housing, loan, contact,
                                             month, day_of_week, duration, campaign, pdays,
                                             previous, poutcome)
        except Exception as e:
            st.error(f"La prédiction a échoué. Vérifiez les valeurs saisies. Détail : {e}")
        else:
            positive = label.lower() in POSITIVE
            title = "Souscription probable" if positive else "Souscription peu probable"
            conf = f" · confiance {proba[label] * 100:.1f} %" if proba and label in proba else ""
            st.markdown(
                f"""
                <div class="result-card {'pos' if positive else 'neg'}" role="status">
                  <p class="result-label">Prédiction du modèle</p>
                  <p class="result-value">{title}</p>
                  <p class="result-note">Classe prédite : {html.escape(label)}{conf}</p>
                </div>
                <div class="chips">
                  <span class="chip">{age} ans</span>
                  <span class="chip">{html.escape(str(job))}</span>
                  <span class="chip">{html.escape(str(marital))}</span>
                  <span class="chip">{html.escape(str(contact))} · {html.escape(str(month))}</span>
                  <span class="chip">Appel : {duration} s</span>
                  <span class="chip">Campagne précédente : {html.escape(str(poutcome))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if proba:
                st.markdown("**Probabilités**")
                for cls, p in sorted(proba.items(), key=lambda kv: -kv[1]):
                    st.progress(min(max(p, 0.0), 1.0), text=f"{cls} : {p * 100:.1f} %")
                st.caption("Les probabilités indiquent le degré de confiance du modèle ; "
                           "elles ne garantissent pas la décision du client.")
    else:
        st.info("Renseignez le profil du client puis cliquez sur **Prédire**.")

# ----------------------------------------------------------------------------
# Onglet 2 : prédiction multiple (CSV)
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("Prédisez la souscription de plusieurs clients d'un coup : importez un fichier CSV, "
                "puis téléchargez les résultats.")

    with st.container(border=True):
        st.markdown('<div class="section-title">1. Préparer le fichier</div>'
                    '<div class="section-hint">Le fichier doit contenir 14 colonnes, dans cet ordre :</div>',
                    unsafe_allow_html=True)
        st.code(", ".join(COLUMNS), language=None)

        template = pd.DataFrame([[40, uniques[0][0], uniques[1][0], uniques[2][0], uniques[3][0],
                                  uniques[4][0], uniques[5][0], uniques[6][0], uniques[7][0],
                                  180, 1, 999, 0, uniques[8][0]]], columns=COLUMNS)
        st.download_button("Télécharger un fichier modèle", template.to_csv(index=False).encode("utf-8"),
                           file_name="modele_clients.csv", mime="text/csv")

        st.markdown('<div class="section-title">2. Importer le fichier</div>', unsafe_allow_html=True)
        file = st.file_uploader("Fichier CSV", type=["csv"], label_visibility="collapsed")

    if file is None:
        st.session_state.pop("df_pred", None)
        st.info("Importez un fichier CSV pour afficher un aperçu et lancer les prédictions.")
    else:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Impossible de lire ce fichier CSV. Détail : {e}")
            st.stop()

        if df.shape[1] < 14:
            st.error(f"Le fichier contient {df.shape[1]} colonne(s) ; 14 sont attendues. "
                     "Téléchargez le fichier modèle pour vérifier le format.")
        else:
            st.success(f"Fichier chargé : {len(df):,} client(s).")
            if list(df.columns[:14]) != COLUMNS:
                st.warning("Les noms de colonnes diffèrent du modèle. Les 14 premières colonnes "
                           "seront utilisées dans l'ordre indiqué ci-dessus.")
            st.markdown("**Aperçu des données**")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Lancer les prédictions", type="primary"):
                with st.spinner("Calcul des prédictions…"):
                    st.session_state["df_pred"] = Pred_func_csv(df)

            df_pred = st.session_state.get("df_pred")
            if df_pred is not None and len(df_pred) == len(df):
                ok = df_pred["etat"].dropna()
                n_pos = int(ok.astype(str).str.lower().isin(POSITIVE).sum())
                n_err = int(df_pred["etat"].isna().sum())

                if n_err:
                    st.warning(f"{n_err} ligne(s) n'ont pas pu être prédites (valeur manquante ou "
                               "catégorie inconnue). Elles apparaissent vides dans la colonne `etat`.")
                else:
                    st.success("Toutes les prédictions ont été calculées.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Clients analysés", f"{len(df_pred):,}")
                m2.metric("Souscriptions probables", f"{n_pos:,}")
                m3.metric("Peu probables", f"{len(ok) - n_pos:,}")

                st.markdown("**Résultats**")
                st.dataframe(df_pred, use_container_width=True)
                st.download_button("Télécharger les résultats (CSV)",
                                   df_pred.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv", mime="text/csv", type="primary")

# ----------------------------------------------------------------------------
# Onglet 3 : à propos
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("### Comment fonctionne la prédiction ?")
    st.write(
        "Un modèle de machine learning (XGBoost) a appris, à partir de campagnes passées, quels profils "
        "de clients souscrivent le plus souvent à un dépôt à terme. Il utilise 14 variables : profil du "
        "client, situation financière et informations sur le contact et les campagnes précédentes."
    )
    st.markdown("**Limites à connaître**")
    st.write(
        "- Le résultat est une **aide à la décision**, pas une certitude.\n"
        "- La **durée de l'appel** n'est connue qu'après l'appel : le modèle sert à analyser des contacts "
        "déjà réalisés plutôt qu'à cibler des clients avant de les appeler.\n"
        "- Le modèle ne connaît que les catégories vues à l'entraînement.\n"
        "- Les valeurs très éloignées des données d'entraînement donnent des résultats moins fiables."
    )
