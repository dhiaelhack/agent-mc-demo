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

**Il n'a pas été testé avec un vrai LLM.** Le planificateur testé, `RuleBasedPlanner`, simule un LLM par des
règles. Le planificateur `OllamaPlanner`, qui appelle un LLM open source local via Ollama, est écrit mais non
testé ici, car je n'avais pas de modèle disponible.

## Contenu

- `mc_slab.py` : simulation Monte Carlo (tirage exponentiel de la distance avant interaction), recherche d'épaisseur
  par dichotomie, comparaison à la solution analytique.
- `agent.py` : boucle d'agent, outils `run_simulation` et `find_thickness`, traçage JSON dans `traces/run.jsonl`.
- `tests.py` : 4 tests (accord avec l'analytique, dichotomie, détection du matériau, gestion des erreurs d'outil).

## Utilisation

```bash
pip install -r requirements.txt
#   ollama pull <modèle>   puis   python agent.py "..." --planner ollama --model <modèle>
```

## Un défaut trouvé en cours de route

Mon simulateur de LLM détectait « eau » dans le mot « faisceau » et ajoutait de l'eau aux matériaux demandés.
Corrigé et couvert par un test. Cela illustre le point qui m'intéresse : évaluer un agent sur des cas tests
précis, et pas seulement sur sa réponse finale.

## Suite que je proposerais

- Brancher un vrai code Monte Carlo (génération du fichier d'entrée, lancement, lecture des résultats).
- Reproduire les exemples de la publication de référence et comparer plusieurs LLM open source.
- Définir des cas tests représentatifs des tâches du laboratoire, avec un critère de réussite mesurable
  (bon outil appelé, bons paramètres, bon résultat).
