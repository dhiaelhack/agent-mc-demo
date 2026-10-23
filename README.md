# Démo : un agent à outils qui pilote un calcul Monte Carlo (version minimale)

Auteur : Dhia Elhack NSIBI (ENISo, Tunisie) – démo préparée en lien avec le sujet de stage
« Utilisation d'agents IA pour automatiser des calculs Monte Carlo » (LMN).

## Ce que c'est, et ce que ce n'est pas

Une boucle d'agent à outils très simple : une demande en langage naturel est transformée en appels
d'outils (simulations), chaque appel est tracé, les erreurs sont renvoyées à l'agent, et la réponse finale
s'appuie sur les résultats des simulations.

**Ce n'est pas** un pilotage d'un vrai code de transport de particules (MCNP, Geant4, TRIPOLI…). Le « calcul
Monte Carlo » est un jouet (transmission non diffusée de photons de 662 keV à travers une plaque, sans diffusion
ni build-up), et ses coefficients d'atténuation sont approximatifs. Il sert de cible pour tester la boucle d'agent.

**Test avec un vrai LLM (un seul essai manuel).** Configuration : openSUSE, Ollama avec le module Vulkan,
RTX 3050 (4 Gio), modèle `llama3.2:3b` chargé à 100 % sur la carte. Sur la demande « Quelle épaisseur de plomb
pour réduire le faisceau à 1 % ? », l'agent a appelé `find_thickness` avec le bon matériau et la bonne cible
(3,649 cm contre 3,655 cm en analytique). Deux exécutions ont montré des défauts dans la trace : la réponse finale
était un objet JSON et non une phrase, et la cible était passée tantôt en nombre (`0.01`), tantôt en chaîne
(`"0.01"`). Par lecture du code, j'ai aussi vu que rien n'empêchait un modèle d'écrire `1` pour 1 %, ce qui donnerait
une épaisseur nulle sans erreur. Tout cela est corrigé (voir ci-dessous). **Un seul essai ne prouve rien** :
d'où le banc d'évaluation.

## Contenu

- `mc_slab.py` : simulation Monte Carlo (tirage exponentiel de la distance avant interaction), recherche d'épaisseur
  par dichotomie, comparaison à la solution analytique.
- `agent.py` : boucle d'agent, outils `run_simulation` et `find_thickness` (arguments validés), nouvel essai quand le
  LLM renvoie un JSON invalide, traçage JSON dans `traces/run.jsonl`.
- `cases.json` : 12 demandes avec les appels attendus (jeu A, utilisé pour concevoir les prompts).
- `cases_holdout.json` : 10 demandes écrites après le prompt v3 et jamais utilisées pour le régler (jeu B, témoin).
- `eval.py` : lance chaque cas et mesure appels corrects, valeurs correctes, sorties invalides, erreurs d'outil,
  arrêts à 8 étapes et durée ; garde la trace complète de chaque exécution.
- `report.py` : relit un fichier de résultats pour détailler les échecs ou re-noter avec les règles actuelles.
- `tests.py` : 16 tests (simulation, dichotomie, validation des arguments, nouvel essai sur JSON invalide, garde-fous,
  détection d'une réponse tronquée, notation du banc d'évaluation, jeu B bien formé et disjoint du jeu A).

## Utilisation

Créer l'environnement virtuel et installer les dépendances :

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Lancer les tests et essayer l'agent avec le planificateur à règles :

```bash
.venv/bin/python tests.py
.venv/bin/python agent.py "Quelle épaisseur de plomb pour réduire le faisceau à 1 % ?"
```

Pour tester le planificateur avec un vrai LLM, installer [Ollama](https://ollama.com/download/linux), télécharger
un modèle (par exemple `ollama pull llama3.2:3b`, qui tient sur une carte de 4 Gio ; vérifier la licence du modèle), puis lancer :

```bash
.venv/bin/python agent.py "Quelle épaisseur de plomb pour réduire le faisceau à 1 % ?" \
  --planner ollama --model llama3.2:3b
```

Lancer le banc d'évaluation :

```bash
.venv/bin/python eval.py --planner rule                                   # base de comparaison à règles
.venv/bin/python eval.py --planner ollama --model llama3.2:3b --runs 1 --temperature 0 --prompt v3
.venv/bin/python eval.py --planner ollama --model llama3.2:3b --runs 1 --temperature 0 --prompt v3 --cases cases_holdout.json
.venv/bin/python eval.py ... --guardrails --show-failures                # options
.venv/bin/python report.py results/<fichier>.json --rescore              # re-noter sans relancer le modèle
```

## Un défaut trouvé en cours de route

Mon simulateur de LLM détectait « eau » dans le mot « faisceau » et ajoutait de l'eau aux matériaux demandés.
Corrigé et couvert par un test. Cela illustre le point qui m'intéresse : évaluer un agent sur des cas tests
précis, et pas seulement sur sa réponse finale.

## Évaluation

Critère de réussite d'une exécution : bons outils appelés avec les bons arguments, et valeurs correctes dans la
réponse finale (tolérance de 1 % ou 0,05 cm pour les épaisseurs, 2 % pour les transmissions). Pour les demandes
irréalisables, réussite = aucun outil n'a abouti, l'agent a bien conclu (il ne s'est pas arrêté à la limite de 8
étapes) et la réponse ne donne aucune épaisseur (critère heuristique). Le simulateur à règles sert de plancher : 6/12 sur le jeu A.

Modèle : `llama3.2:3b` via Ollama (paquets openSUSE `ollama` et `ollama-vulkan`), RTX 3050 (4 Gio).
Température 0, une exécution par cas, `max_tokens` = 400, délai = 120 s.

- **Jeu A** (12 cas) : a servi à concevoir les prompts, donc non indépendant.
- **Jeu B** (10 cas) : écrit après le prompt v3 et non utilisé pour le régler.

| Configuration | Jeu A, réussites | Jeu B, réussites | Erreurs d'outil (A / B) |
|---|---|---|---|
| prompt v1 | 6/12 | 6/10 | environ 15 / 18 |
| prompt v2 | 7/12 puis 6/12 (message d'erreur modifié entre les deux lancements) | non mesuré | 7 puis 8 / non mesuré |
| prompt v3 | 8/12 | 7/10 | 3 / 2 |
| prompt v3 + garde-fous | non mesuré | 8/10 | non mesuré / 2 |

Lecture, avec les réserves :

- **Solide** : les exemples du prompt réduisent nettement les appels d'outil mal formés, sur les deux jeux de cas.
- **Non démontré** : un gain de réussites. Les écarts sont d'un cas sur 10 ou 12, et un score de 7/10 est
  compatible avec un vrai taux entre 40 % et 90 %.
- **Échec persistant** : le modèle ne refuse pas de façon fiable une demande impossible (matériau absent,
  transmission hors limites) ; il remplace la valeur par une autre. Les garde-fous du code règlent le matériau
  inconnu, pas la transmission hors limites ; ils mesurent donc le code, pas le modèle.
- **Variabilité** : à température 0,7, deux lancements du prompt v1 ont donné 14/36 puis 13/36. À température 0,
  trois exécutions d'un même cas donnent le même résultat, et trois lancements espacés dans le temps du prompt v1,
  puis du prompt v3, sur le jeu B ont donné chacun le même score (6/10 après correction, puis 7/10). Je n'ai pas
  vérifié que les réponses sont identiques cas par cas.
- **Correction de méthode** : le critère de réussite des refus a été corrigé après coup, car un agent qui épuise
  ses 8 étapes sans répondre comptait à tort comme un refus réussi. Les résultats du jeu B ont été re-notés ; la
  règle corrigée ne peut que retirer des réussites, et celles du jeu A sont des demandes ordinaires dont la
  réponse contient les bonnes valeurs, donc inchangées.
- **Limites** : un seul modèle de 3 milliards de paramètres, une exécution par cas, des jeux de cas écrits par moi
  et un critère de refus heuristique.

### Fichiers de résultats

Chaque chiffre du tableau correspond à un fichier de `results/` (préfixe commun : `eval_ollama_llama3.2-3b_`).
Les scores enregistrés dans un fichier sont ceux de la règle de notation au moment du lancement ;
`python report.py <fichier> --rescore` donne les scores corrigés (v1, jeu B : 7/10 enregistré, 6/10 corrigé).

| Ligne du tableau | Fichier(s) |
|---|---|
| v1, jeu A (3 exécutions identiques, T = 0) | `v1_20261022-194033.json` |
| v2, jeu A | `v2_20261022-200334.json` (7/12) ; `v2_20261022-200743.json` (6/12) |
| v3, jeu A | `v3_20261022-200952.json` |
| v1, jeu B | `v1_20261022-201830.json`, `v1_20261023-112124.json`, `v1_20261023-112430.json` |
| v3, jeu B | `v3_20261022-201759.json`, `v3_20261022-203405.json`, `v3_20261023-112227.json` |
| v3 + garde-fous, jeu B | `v3_garde_20261023-112555.json` |
| Température 0,7, jeu A, prompt v1 | `20261022-192940.json` (14/36, sans trace) ; `20261022-193414.json` (13/36) |
| Simulateur à règles, jeu A | `eval_rule_20261022-192601.json` (sans trace) |

## Suite que je proposerais

- Brancher un vrai code Monte Carlo (génération du fichier d'entrée, lancement, lecture des résultats).
- Reproduire les exemples de la publication de référence et comparer plusieurs LLM open source.
- Définir des cas tests représentatifs des tâches du laboratoire, avec un critère de réussite mesurable
  (bon outil appelé, bons paramètres, bon résultat).
