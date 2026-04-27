# Hybrid Video Alignment (Coarse-to-Fine: DTW + AKAZE)

Alignement temporel de vidéos de trains utilisant une approche coarse-to-fine combinant:
- **Phase Macro (DTW)** : Alignement robuste basé sur le flux optique
- **Phase Micro (AKAZE)** : Raffinement précis par détection de features
- **Phase Fallback** : Filet de sécurité pour les zones difficiles

## Installation

```bash
cd /Users/julian/Documents/Master2/Q2/Master_Thesis/Projet/TSV
```

## Structure du Dataset

```
dataset/
├── Plan1/
│   ├── video1.mp4  (vidéo J-1, référence)
│   ├── video2.mp4  (vidéo J, cible)
│   └── hybrid/     (résultats créés automatiquement)
├── Plan2/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── hybrid/
└── Plan3/
    ├── video1.mp4
    ├── video2.mp4
    └── hybrid/
```

## Utilisation

### Mode Interactif (Recommandé)

```bash
# Lance le script interactif
python combined_method/run_alignment_hybrid.py
```

Affiche les plans disponibles et demande de sélectionner :

```
======================================================================
HYBRID VIDEO ALIGNMENT (Coarse-to-Fine DTW + AKAZE)
======================================================================

📁 Available Plans:
   1. Plan1
   2. Plan2
   3. PlanTest

🎛️  Options:
   a. Process All
   q. Quit

➜ Select plan number or option:
```

### Mode Automatisé

```bash
# Traiter un plan spécifique
python combined_method/run_alignment_hybrid.py Plan1

# Traiter tous les plans
python combined_method/run_alignment_hybrid.py all
```

## Outputs

Pour chaque plan, le dossier `dataset/Plan1/hybrid/` contient :

### 📹 Vidéos alignées (3 modes)
- `aligned_simple.mp4` - Side-by-side simple (le plus rapide)
- `aligned_features.mp4` - Avec keypoints AKAZE affichés
- `aligned_flow.mp4` - Avec flux optique visualisé

### 📊 Visualisations
- `alignment_scatter.png` - Graphique V1 vs V2 frames
- `alignment_difference.png` - Corrections AKAZE par rapport au DTW
- `source_distribution.png` - Pie chart + qualité AKAZE
- `velocity_analysis.png` - Vitesse relative entre vidéos
- `dtw_cost_matrix.png` - Heatmap DTW avec chemin optimal

### 📈 Données structurées
- `alignment_results.csv` - Correspondances frame par frame
- `metrics.json` - Métriques de qualité

### 📄 Rapport
- `report.html` - Rapport HTML complet (ouvrir dans un navigateur)

## Métriques clés

| Métrique | Description |
|----------|-------------|
| **Total matches** | Nombre total de correspondances trouvées |
| **AKAZE refined %** | % de frames affinées par AKAZE (bon signe !) |
| **DTW fallback %** | % où DTW a assuré la sécurité |
| **Monotonicity score** | % de respect de progression monotone (progression normale) |
| **Velocity ratio** | Rapport de vitesse moyen (1.0 = même vitesse) |
| **Mean AKAZE inliers** | Qualité moyenne des matchs AKAZE |
| **Max correction** | Plus grande correction apportée par AKAZE |

## API Python

```python
from combined_method import align_videos_hybrid

# Utilisation simple
results = align_videos_hybrid(
    "video1.mp4",
    "video2.mp4",
    dtw_sample_rate=5,      # Extraction rapide (1 frame sur 5)
    akaze_sample_rate=1,    # Raffinement sur toutes les frames
    search_window=10,       # Fenêtre stricte ±10 frames
    min_inliers=4           # Seuil de fallback
)

# Résultat: liste de dictionnaires
for match in results:
    print(f"V1[{match['v1_frame']}] -> V2[{match['v2_frame']}] "
          f"(source: {match['source']}, inliers: {match['score']})")
```

## Paramètres ajustables

### Pour plus de précision (mais plus lent)
```bash
python combined_method/run_alignment_hybrid.py Plan1 \
    --dtw-sample-rate 2 \
    --akaze-sample-rate 1 \
    --search-window 15
```

### Pour plus de rapidité (moins précis)
```bash
python combined_method/run_alignment_hybrid.py Plan1 \
    --dtw-sample-rate 10 \
    --akaze-sample-rate 5 \
    --search-window 8
```

## Architecture

```
combined_method/
├── __init__.py              # Exports principaux
├── hybrid_alignment.py      # Cœur de l'algorithme Coarse-to-Fine
├── visualization.py         # Génération de visualisations
├── run_alignment_hybrid.py  # ← SCRIPT À LANCER
└── run_hybrid.py            # Alternative (CLI avancée)
```

## Troubleshooting

### "No plans found in 'dataset'"
- Vérifiez que vous êtes dans le bon répertoire
- Le dossier `dataset/` doit être au même niveau que `combined_method/`

### Les vidéos sont très grandes
- Utilisez des `sample_rate` plus élevés lors de l'extraction DTW
- Limitez avec `--max-video-frames`

### Qualité faible (trop de DTW fallback)
- Réduisez `--search-window` pour forcer AKAZE à être plus précis
- Augmentez `--min-inliers` pour être plus sélectif

## Citation

Approche Coarse-to-Fine utilisant :
- **DTW (Dynamic Time Warping)** : Flux optique dense pour l'alignement global robuste
- **AKAZE** : Détection rapide de features avec RANSAC pour le raffinement

---

**Questions ? Consultez la documentation détaillée ou ouvrez les rapports HTML générés.**
