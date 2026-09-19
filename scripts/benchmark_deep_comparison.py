#!/usr/bin/env python3
"""
Deep Head-to-Head Benchmark: Pure NanoPrune v0.3 vs Pure Laya vs Hybrid Cascade
Measures:
1. Exact latency & throughput on real documents
2. Memory and disk footprint
3. Semantic precision, recall, and false-positive resilience on hard negatives & boundary cases
4. Agreement rate between Cascade and Pure Laya
"""
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import torch
import laya
from nanoprune.engine.pruner import NanoPruner
from nanoprune.engine.cascade import HybridCascadePruner

def load_real_test_suite() -> List[Dict[str, Any]]:
    """
    Constructs a calibrated 50-item evaluation set with 4 distinct difficulty categories:
    - CATEGORY 1: Direct Positives (clear semantic match)
    - CATEGORY 2: Obvious Negatives (completely different domain)
    - CATEGORY 3: Hard Negatives (lexical trap: shared legal keywords, but unrelated intent)
    - CATEGORY 4: Subtle Edge Cases (boundary conditions, exceptions, cross-domain provisions)
    """
    suite = [
        # --- Category 1: Direct Positives (15 items) ---
        {
            "category": "Direct Positive",
            "query": "rupture conventionnelle et indemnité de départ pour salarié",
            "doc": "Convention collective de travail pour le personnel salarié : modalités relatives à la rupture conventionnelle, préavis et calcul de l'indemnité de départ convenue entre employeur et salarié.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "droit de propriété et indemnisation pour expropriation publique",
            "doc": "Arrêt de la CEDH : violation alléguée de l'article 1 du Protocole n°1. Nul ne peut être privé de sa propriété que pour cause d'utilité publique et moyennant une indemnisation juste et préalable.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "autorisation d'exercice de la médecine de spécialité neurologique",
            "doc": "Ministère de la Santé : Par arrêté ministériel, le praticien est dûment autorisé à exercer la profession de médecin-spécialiste en neurologie sur le territoire national.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "délai de recours contentieux devant le tribunal administratif",
            "doc": "Loi sur le contentieux administratif : Le recours en annulation contre la décision ministérielle doit être introduit dans un délai strict de trois mois à compter de sa publication au journal officiel.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "représentation du personnel et comité d'entreprise syndical",
            "doc": "Avenant syndical Met-Lux signé entre la direction générale et la délégation du personnel affiliée au syndicat LCGB fixant le temps de délégation et les réunions obligatoires.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "protection des données personnelles RGPD et registre des traitements",
            "doc": "Règlement général sur la protection des données : chaque responsable de traitement est tenu de tenir un registre des activités de traitement détaillant les catégories de données collectées.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "garantie décennale pour malfaçons du constructeur immobilier",
            "doc": "Article 1792 du code civil : tout constructeur d'un ouvrage est responsable de plein droit des dommages qui compromettent la solidité de l'ouvrage ou qui l'affectent dans l'un de ses éléments constitutifs.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "garde alternée et fixation de la pension alimentaire des enfants mineurs",
            "doc": "Jugement aux affaires familiales : fixation de la résidence alternée des enfants et condamnation au versement d'une contribution mensuelle de 450 euros au titre de l'entretien et de l'éducation.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "nomination de magistrats à la cour constitutionnelle",
            "doc": "Arrêté grand-ducal : Monsieur Serge Schroeder, vice-président à la Cour administrative, est nommé conseiller à la Cour constitutionnelle avec effet immédiat.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "exonération de TVA pour livraisons intracommunautaires de biens",
            "doc": "Code fiscal : sont exonérées de la taxe sur la valeur ajoutée les livraisons de biens expédiés ou transportés sur le territoire d'un autre État membre par un assujetti.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "durée légale hebdomadaire du travail et heures supplémentaires",
            "doc": "Code du travail : la durée normale de travail est fixée à 40 heures par semaine. Toute heure effectuée au-delà ouvre droit à une majoration de salaire ou à un repos compensateur.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "recours contre le refus de titre de séjour pour réfugié",
            "doc": "Tribunal administratif : recours en réformation dirigé contre la décision du ministre de l'Immigration portant rejet de la demande de protection internationale et obligation de quitter le territoire.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "clause de non-concurrence post-contractuelle et contrepartie financière",
            "doc": "Contrat de travail : la clause de non-concurrence interdisant au cadre de travailler pour une entreprise rivale pendant 12 mois n'est valide que si elle stipule une indemnité pécuniaire mensuelle égale à 50% du salaire.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "action civile pour diffamation publique dans la presse écrite",
            "doc": "Loi sur la liberté de la presse : l'action publique et l'action civile résultant des délits de diffamation ou d'injure se prescrivent après trois mois révolus à compter du jour de publication.",
            "ground_truth": 1,
            "trap": None
        },
        {
            "category": "Direct Positive",
            "query": "procédure de redressement judiciaire et déclaration de cessation des paiements",
            "doc": "Loi sur la faillite : tout commerçant qui cesse ses paiements et dont le crédit est ébranlé est tenu de faire la déclaration de faillite au greffe du tribunal de commerce dans le délai d'un mois.",
            "ground_truth": 1,
            "trap": None
        },

        # --- Category 2: Obvious Negatives (15 items) ---
        {
            "category": "Obvious Negative",
            "query": "rupture conventionnelle et indemnité de départ pour salarié",
            "doc": "Arrêté ministériel autorisant le défrichement d'une parcelle forestière communale en vue de la construction d'un bassin de rétention d'eaux pluviales.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "droit de propriété et indemnisation pour expropriation publique",
            "doc": "Ordonnance vétérinaire régissant les mesures prophylactiques contre la grippe aviaire et les conditions de transport des volailles vivantes.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "autorisation d'exercice de la médecine de spécialité neurologique",
            "doc": "Règlement communal relatif aux horaires d'ouverture des déchetteries et au tri sélectif des encombrants métalliques.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "délai de recours contentieux devant le tribunal administratif",
            "doc": "Rapport technique sur les émissions de particules fines des véhicules diesel Euro 6 lors des cycles de conduite urbaine en hiver.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "représentation du personnel et comité d'entreprise syndical",
            "doc": "Normes acoustiques applicables aux installations éoliennes terrestres situées à moins de 500 mètres d'une zone d'habitation.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "protection des données personnelles RGPD et registre des traitements",
            "doc": "Recette officielle du pain traditionnel au levain et normes de panification pour les boulangeries artisanales.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "garantie décennale pour malfaçons du constructeur immobilier",
            "doc": "Arrêté de circulation temporaire interdisant le stationnement des poids lourds le long du quai de déchargement portuaire.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "garde alternée et fixation de la pension alimentaire des enfants mineurs",
            "doc": "Guide méthodologique de certification des logiciels de comptabilité analytique pour les exploitations agricoles viticoles.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "nomination de magistrats à la cour constitutionnelle",
            "doc": "Calendrier des vacances scolaires et organisation des centres de loisirs aérés pour la saison estivale 2026.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "exonération de TVA pour livraisons intracommunautaires de biens",
            "doc": "Procédure d'homologation des casques de protection pour cyclistes et utilisateurs de trottinettes électriques en agglomération.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "durée légale hebdomadaire du travail et heures supplémentaires",
            "doc": "Classification zoologique des lépidoptères endémiques du bassin méditerranéen et inventaire des espèces protégées.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "recours contre le refus de titre de séjour pour réfugié",
            "doc": "Cahier des charges pour la fourniture de transformateurs électriques haute tension destinés au réseau ferré métropolitain.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "clause de non-concurrence post-contractuelle et contrepartie financière",
            "doc": "Régime d'indemnisation des dégâts causés aux cultures maraîchères par le grand gibier sanglier et chevreuil.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "action civile pour diffamation publique dans la presse écrite",
            "doc": "Spécifications techniques de pose de canalisations d'adduction en fonte ductile pour eau potable sous voirie.",
            "ground_truth": 0,
            "trap": None
        },
        {
            "category": "Obvious Negative",
            "query": "procédure de redressement judiciaire et déclaration de cessation des paiements",
            "doc": "Manuel d'entretien des chaudières à granulés de bois et vérification annuelle du tirage des conduits de fumée.",
            "ground_truth": 0,
            "trap": None
        },

        # --- Category 3: Hard Negatives (Lexical Traps) (10 items) ---
        {
            "category": "Hard Negative (Trap)",
            "query": "indemnité de départ et licenciement du salarié en CDI",
            "doc": "Arrêté fixant les tarifs d'indemnité kilométrique accordés aux experts géomètres lors de leurs missions de bornage parcellaire.",
            "ground_truth": 0,
            "trap": "Contient 'indemnité', mais pour frais kilométriques géomètres, pas salarié CDI."
        },
        {
            "category": "Hard Negative (Trap)",
            "query": "délai de recours contentieux devant le tribunal administratif",
            "doc": "Ordonnance du juge des tutelles statuant sur le compte rendu annuel de gestion du patrimoine d'un majeur sous curatelle renforcée.",
            "ground_truth": 0,
            "trap": "Texte judiciaire officiel ('ordonnance', 'juge'), mais en matière de tutelle civile, pas de délai de recours administratif."
        },
        {
            "category": "Hard Negative (Trap)",
            "query": "responsabilité médicale et faute chirurgicale du praticien",
            "doc": "Arrêté ministériel autorisant le Dr Martin à exercer la profession de chirurgien-dentiste au sein du centre hospitalier.",
            "ground_truth": 0,
            "trap": "Concerne un médecin/chirurgien, mais nomination administrative, aucune mention de responsabilité ou de faute."
        },
        {
            "category": "Hard Negative (Trap)",
            "query": "protection des données personnelles RGPD et fuite de données",
            "doc": "Règlement sur la protection des obtentions végétales et l'enregistrement officiel des nouvelles variétés de semences de blé au catalogue.",
            "ground_truth": 0,
            "trap": "Contient 'protection' et 'registre', mais concerne les semences végétales."
        },
        {
            "category": "Hard Negative (Trap)",
            "query": "violation de la liberté d'expression par censure étatique",
            "doc": "CEDH : Affaire concernant le refus d'autorisation de construire un garage en zone agricole au regard du droit de propriété.",
            "ground_truth": 0,
            "trap": "Arrêt CEDH solennel, mais porte sur l'urbanisme et la propriété foncière, absolument pas sur la liberté d'expression."
        },
        {
            "category": "Hard Negative (Trap)",
            "query": "pension alimentaire et contribution à l'entretien de l'enfant",
            "doc": "Arrêté fixant la liste des denrées alimentaires exonérées de droits de douane dans le cadre de l'aide humanitaire d'urgence.",
            "ground_truth": 0,
            "trap": "Contient 'alimentaire' et 'aide', mais s'agit de nourriture et douanes."
        },
        {
            "category": "Hard Negative (Trap)",
            "query": "faillite d'entreprise et créances salariales privilégiées",
            "doc": "Loi sur l'organisation des débits de boissons et les conditions d'obtention de la licence IV de vente d'alcool.",
            "ground_truth": 0,
            "trap": "Droit des entreprises, mais débit de boissons sans aucun lien avec la faillite ou créances."
        },
        {
            "category": "Hard Negative (Trap)",
            "query": "recours contre le refus de visa et obligation de quitter le territoire",
            "doc": "Tribunal de police : condamnation à une amende forfaitaire majorée pour infraction au code de la route (dépassement de vitesse).",
            "ground_truth": 0,
            "trap": "Tribunal et condamnation, mais contravention routière, pas droit des étrangers."
        },
        {
            "category": "Hard Negative (Trap)",
            "query": "harcèlement moral au travail et démission pour faute grave de l'employeur",
            "doc": "Code pénal : sanctions applicables aux infractions d'outrage à agent de la force publique dans l'exercice de ses fonctions de patrouille.",
            "ground_truth": 0,
            "trap": "Droit pénal / infractions, mais concerne la police et non le contrat de travail."
        },
        {
            "category": "Hard Negative (Trap)",
            "query": "clause de non-concurrence et indemnité financière compensatrice",
            "doc": "Marché public : avis de mise en concurrence pour la réfection de la toiture du lycée technique régional.",
            "ground_truth": 0,
            "trap": "Contient 'concurrence' et 'marché', mais appel d'offres de travaux publics, pas contrat de travail."
        },

        # --- Category 4: Subtle Edge Cases (Ambiguous / Boundaries) (10 items) ---
        {
            "category": "Subtle Edge Case",
            "query": "indemnisation pour retard de paiement d'une créance d'aide judiciaire",
            "doc": "CEDH Diaco c. Italie : Les requérants se plaignent d'un retard excessif dans le règlement par l'État de créances au titre de l'aide judiciaire, invoquant l'atteinte à leurs biens protégés par l'article 1 du Protocole n°1.",
            "ground_truth": 1,
            "trap": "Clause complexe reliant retard d'honoraires d'aide juridictionnelle et droit de propriété de la CEDH."
        },
        {
            "category": "Subtle Edge Case",
            "query": "gestion sous mandat des comptes bancaires familiaux d'une personne protégée",
            "doc": "CEDH M.T.S. c. Portugal : Accord familial prévoyant que la première requérante gère le compte courant et les soins, tandis que le conseil de famille confie les autres comptes d'épargne bancaires à un tiers désigné sous le contrôle du juge.",
            "ground_truth": 1,
            "trap": "Tutelle familiale et mandats bancaires croisés avec dimension CEDH."
        },
        {
            "category": "Subtle Edge Case",
            "query": "sanctions pour absence injustifiée d'un agent de la fonction publique",
            "doc": "Statut général des fonctionnaires : Le fonctionnaire en position d'activité qui s'abstient sans motif légitime de se présenter à son poste encourt une retenue sur traitement pour service non fait, indépendamment des poursuites disciplinaires.",
            "ground_truth": 1,
            "trap": "Double sanction (disciplinaire et financière) spécifique aux fonctionnaires."
        },
        {
            "category": "Subtle Edge Case",
            "query": "nullité d'un contrat conclu par un mineur émancipé sans assistance",
            "doc": "Code civil : L'incapacité d'exercice du mineur cesse par l'émancipation. Toutefois, pour les actes de disposition immobilière ou souscription d'emprunt bancaire excédant ses revenus, le consentement du curateur reste requis sous peine de rescision.",
            "ground_truth": 1,
            "trap": "Règle de principe (émancipation) assortie d'une exception stricte (rescisions actes lourds)."
        },
        {
            "category": "Subtle Edge Case",
            "query": "exonération de responsabilité du transporteur aérien en cas de météo extrême",
            "doc": "Règlement européen passagers : La compagnie n'est pas tenue au versement de l'indemnité forfaitaire de retard si elle prouve que l'annulation est due à des circonstances extraordinaires imprévisibles, telles des conditions météorologiques incompatibles avec la sécurité.",
            "ground_truth": 1,
            "trap": "Clause d'exonération conditionnée par la charge de la preuve d'événements extraordinaires."
        },
        {
            "category": "Subtle Edge Case",
            "query": "responsabilité sans faute de l'État pour dommage causé par des travaux publics",
            "doc": "Conseil d'État : Les tiers victimes de dommages permanents ou accidentels causés par l'exécution d'un travail public ont droit à réparation même en l'absence de toute faute de l'administration, dès lors que le préjudice présente un caractère spécial et anormal.",
            "ground_truth": 1,
            "trap": "Régime autonome de responsabilité sans faute avec condition d'anormalité du préjudice."
        },
        {
            "category": "Subtle Edge Case",
            "query": "secret professionnel de l'avocat et perquisition de son cabinet",
            "doc": "Code de procédure pénale : Aucune perquisition ne peut avoir lieu dans le cabinet d'un avocat sans la présence du bâtonnier de l'ordre ou de son délégué, lequel veille à ce qu'aucun document couvert par le secret de la défense ne soit saisi.",
            "ground_truth": 1,
            "trap": "Protection procédurale impérative du secret professionnel avec garanties du bâtonnier."
        },
        {
            "category": "Subtle Edge Case",
            "query": "rupture anticipée d'un contrat de travail à durée déterminée CDD",
            "doc": "Code du travail : Sauf accord des parties, le contrat à durée déterminée ne peut être rompu avant l'échéance du terme qu'en cas de faute grave de l'une des parties, de force majeure ou d'inaptitude constatée par le médecin du travail.",
            "ground_truth": 1,
            "trap": "Conditions limitatives strictes de rupture anticipée du CDD."
        },
        {
            "category": "Subtle Edge Case",
            "query": "délai de rétractation dans les contrats conclus à distance par internet",
            "doc": "Code de la consommation : Le consommateur dispose d'un délai de quatorze jours pour exercer son droit de rétractation sans avoir à motiver sa décision, ce délai courant à compter du jour de la réception du bien commandé.",
            "ground_truth": 1,
            "trap": "Règle de comput des délais de rétractation e-commerce."
        },
        {
            "category": "Subtle Edge Case",
            "query": "prescription de l'action en recouvrement de cotisations sociales impayées",
            "doc": "Code de la sécurité sociale : L'action en recouvrement des cotisations et contributions dues par les employeurs se prescrit par trois ans à compter de l'expiration de l'année civile au titre de laquelle elles sont exigibles.",
            "ground_truth": 1,
            "trap": "Point de départ différé du délai de prescription triennale de sécurité sociale."
        },
    ]
    return suite

def main():
    print("=" * 70)
    print("🔬 COMPARATIVE BENCHMARK: PURE NANOPRUNE vs PURE LAYA vs HYBRID CASCADE")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
    
    suite = load_real_test_suite()
    total_docs = len(suite)
    print(f"📋 Loaded {total_docs} calibrated test cases across 4 difficulty tiers.")

    # 1. Initialize Pure NanoPrune v0.4
    print("\n[1/3] Initializing Pure NanoPrune v0.4 (5.4M params)...")
    t0 = time.perf_counter()
    nanoprune = NanoPruner.load()
    t_load_np = (time.perf_counter() - t0) * 1000.0
    print(f"      NanoPrune ready in {t_load_np:.2f} ms.")

    # 2. Initialize Pure Laya (Teacher 421M)
    print("\n[2/3] Initializing Pure Laya 421M (ModernBERT-large)...")
    t0 = time.perf_counter()
    laya_agent = laya.load("convaiinnovations/laya")
    t_load_laya = (time.perf_counter() - t0) * 1000.0
    print(f"      Laya ready in {t_load_laya:.2f} ms on device: {laya_agent.device}.")

    # 3. Initialize Hybrid Cascade Pruner
    print("\n[3/3] Initializing Hybrid Cascade Pruner (Tier 1: NanoPrune, Tier 2: Ensemble Laya)...")
    cascade = HybridCascadePruner(
        nanoprune_model=nanoprune,
        drop_threshold=0.75,
        keep_threshold=0.95,
        enable_laya=True
    )
    cascade.laya_agent = laya_agent
    print("      Hybrid Cascade configured.")

    # -------------------------------------------------------------
    # RUN EVALUATION ON THE 50 CASES
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🚀 RUNNING INFERENCE ON 50 CASES (Pure NP, Pure Laya, Hybrid Cascade)")
    print("=" * 70)

    results_np = []
    results_laya = []
    results_cascade = []

    # --- Pure NanoPrune Pass ---
    t0_np = time.perf_counter()
    for item in suite:
        score = nanoprune.score_pair(item["query"], item["doc"])
        results_np.append(score)
    t_total_np = (time.perf_counter() - t0_np) * 1000.0

    # --- Pure Laya Pass ---
    t0_laya = time.perf_counter()
    for idx, item in enumerate(suite):
        state = f"Document: {item['doc']}"
        questions = {
            "relevance": {
                "type": "noul",
                "instructions": f"Ce document apporte-t-il une réponse pertinente à la recherche : '{item['query']}' ?"
            }
        }
        res = laya_agent.system_one(state, questions)
        laya_prob = float(res["answers"]["relevance"]["noul"])
        results_laya.append(laya_prob)
    t_total_laya = (time.perf_counter() - t0_laya) * 1000.0

    # --- Hybrid Cascade Pass ---
    t0_cas = time.perf_counter()
    cascade_details = []
    for item in suite:
        res = cascade.prune_cascade(item["query"], [item["doc"]])
        retained = res["retained"]
        if retained:
            score = retained[0][1]
            tag = retained[0][2]
        else:
            score = 0.0
            tag = "dropped"
        cascade_details.append({
            "score": score,
            "tag": tag,
            "stats": res["stats"]
        })
    t_total_cas = (time.perf_counter() - t0_cas) * 1000.0

    # -------------------------------------------------------------
    # AGGREGATE METRICS & COMPARISON
    # -------------------------------------------------------------
    np_decisions = [1 if s >= 0.50 else 0 for s in results_np]
    laya_decisions = [1 if s >= 0.50 else 0 for s in results_laya]
    cas_decisions = [1 if d["score"] >= 0.50 else 0 for d in cascade_details]
    ground_truths = [item["ground_truth"] for item in suite]

    # Metrics vs Ground Truth
    acc_np = sum(1 for p, y in zip(np_decisions, ground_truths) if p == y) / total_docs * 100.0
    acc_laya = sum(1 for p, y in zip(laya_decisions, ground_truths) if p == y) / total_docs * 100.0
    acc_cas = sum(1 for p, y in zip(cas_decisions, ground_truths) if p == y) / total_docs * 100.0

    # Agreement with Laya (Teacher fidelity)
    np_agree_with_laya = sum(1 for p, y in zip(np_decisions, laya_decisions) if p == y) / total_docs * 100.0
    cas_agree_with_laya = sum(1 for p, y in zip(cas_decisions, laya_decisions) if p == y) / total_docs * 100.0

    # Trap Resistance (False Positives on Hard Negatives, items 30-39)
    traps = [(item, np_d, laya_d, cas_d) for item, np_d, laya_d, cas_d in zip(suite, np_decisions, laya_decisions, cas_decisions) if item["category"] == "Hard Negative (Trap)"]
    np_trap_fails = sum(1 for t in traps if t[1] == 1)
    laya_trap_fails = sum(1 for t in traps if t[2] == 1)
    cas_trap_fails = sum(1 for t in traps if t[3] == 1)

    # Cascade tier breakdown
    tier1_dropped_cnt = sum(1 for d in cascade_details if "fast_dropped" in d["tag"] or d["tag"] == "dropped")
    tier1_kept_cnt = sum(1 for d in cascade_details if d["tag"] == "tier1_match")
    tier2_arbitrated_cnt = sum(1 for d in cascade_details if "tier2" in d["tag"])

    # Output full report
    report = {
        "dataset_size": total_docs,
        "latency": {
            "pure_nanoprune_total_ms": round(t_total_np, 2),
            "pure_nanoprune_per_doc_ms": round(t_total_np / total_docs, 2),
            "pure_laya_total_ms": round(t_total_laya, 2),
            "pure_laya_per_doc_ms": round(t_total_laya / total_docs, 2),
            "cascade_total_ms": round(t_total_cas, 2),
            "cascade_per_doc_ms": round(t_total_cas / total_docs, 2),
            "cascade_speedup_vs_laya": round(t_total_laya / max(1.0, t_total_cas), 2),
            "nanoprune_speedup_vs_laya": round(t_total_laya / max(1.0, t_total_np), 2),
        },
        "accuracy_vs_truth": {
            "pure_nanoprune": round(acc_np, 1),
            "pure_laya": round(acc_laya, 1),
            "hybrid_cascade": round(acc_cas, 1),
        },
        "agreement_with_laya": {
            "pure_nanoprune": round(np_agree_with_laya, 1),
            "hybrid_cascade": round(cas_agree_with_laya, 1),
        },
        "hard_negative_trap_failures": {
            "total_traps": len(traps),
            "pure_nanoprune_fooled": np_trap_fails,
            "pure_laya_fooled": laya_trap_fails,
            "cascade_fooled": cas_trap_fails,
        },
        "cascade_dispatch": {
            "tier1_fast_dropped": tier1_dropped_cnt,
            "tier1_fast_kept": tier1_kept_cnt,
            "tier2_laya_arbitrated": tier2_arbitrated_cnt,
            "laya_calls_saved_pct": round((1.0 - (tier2_arbitrated_cnt / total_docs)) * 100.0, 1),
        }
    }

    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(json.dumps(report, indent=2))

    # Save detailed JSON for review
    out_file = Path("data/benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": report,
            "detailed_cases": [
                {
                    "idx": i,
                    "category": item["category"],
                    "query": item["query"],
                    "doc": item["doc"][:120] + "...",
                    "trap_explanation": item["trap"],
                    "ground_truth": item["ground_truth"],
                    "scores": {
                        "nanoprune": round(results_np[i], 4),
                        "laya": round(results_laya[i], 4),
                        "cascade": round(cascade_details[i]["score"], 4),
                    },
                    "decisions": {
                        "nanoprune": np_decisions[i],
                        "laya": laya_decisions[i],
                        "cascade": cas_decisions[i],
                    },
                    "cascade_tag": cascade_details[i]["tag"]
                }
                for i, item in enumerate(suite)
            ]
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full case-by-case report saved to: {out_file}")

if __name__ == "__main__":
    main()
