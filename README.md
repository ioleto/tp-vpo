# TP1

Eloi Tourangin et Thomas Verron
SEC3

## 2 - Histogramme

### 2.1 - Image originale

L'image `peppers-512.png` est  en niveaux de gris.

| Image originale | Histogramme original |
|:---:|:---:|
| ![Image originale](imagesDeTest/peppers-512.png) | ![Histogramme original](screen/Figure_1.png) |

L'axe horizontal représente les niveaux de gris, de 0 (noir) à 255 (blanc), et l'axe vertical représente le nombre de pixels. L'histogramme présente principalement deux zones de concentration, autour des niveaux 80 et 180. Les faibles valeurs aux extrémités montrent qu'il y a peu de pixels complètement noirs ou complètement blancs.

### 2.2 - Égalisation de l'histogramme

Transformation utilisée : `cv2.equalizeHist`, sans paramètre supplémentaire.

| Image originale | Image après égalisation |
|:---:|:---:|
| ![Image originale](imagesDeTest/peppers-512.png) | ![Image égalisée](screen/peppers-512-equal.png) |

![Comparaison des histogrammes avant et après égalisation](screen/Figure_2.png)

L'égalisation redistribue les niveaux de gris sur une plage plus large. L'histogramme est donc plus étalé et le contraste augmente, notamment dans les zones où les niveaux étaient initialement proches. La répartition n'est pas parfaitement uniforme, car elle dépend de l'image de départ.

### 2.3 - Modification de la luminosité et du contraste

#### Augmentation de la luminosité

Transformation utilisée : `I' = alpha * I + beta`, avec `alpha=1` et `beta=40`.

| Image originale | Image après augmentation de la luminosité |
|:---:|:---:|
| ![Image originale](imagesDeTest/peppers-512.png) | ![Image plus lumineuse](screen/peppers-512-contrast.png) |

![Comparaison des histogrammes avant et après augmentation de la luminosité](screen/Figure_3.png)

Avec `beta=40`, l'image devient globalement plus claire et l'histogramme se déplace vers les niveaux élevés. Les pixels dépassant 255 sont saturés à 255, ce qui peut supprimer des détails dans les zones très claires.

#### Augmentation du contraste

Transformation utilisée : `I' = alpha * I + beta`, avec `alpha=1.5` et `beta=0`.

| Image originale | Image après augmentation du contraste |
|:---:|:---:|
| ![Image originale](imagesDeTest/peppers-512.png) | ![Image plus contrastée](screen/peppers-512-contrast2.png) |

![Comparaison des histogrammes avant et après augmentation du contraste](screen/Figure_4.png)

Avec `alpha=1.5`, les zones sombres deviennent plus sombres et les zones claires plus claires. L'histogramme s'étale autour de la moyenne. Les valeurs hors de `[0,255]` sont tronquées, ce qui peut provoquer une saturation dans les deux extrêmes.

## 3 - Redimensionnement d’image

### 3.1 - Sous-échantillonnage

L'objectif est de réduire l'image `peppers-256.png` (256×256, niveaux de gris) en une image 64×64, soit un facteur 4 dans chaque direction, et de comparer trois méthodes de réduction.

**Méthode 1 : sélection d'un pixel par bloc 4×4 (double boucle)**

La fonction `sous_echantillonner(image, facteur)` crée une image vide de taille `(h // facteur, w // facteur)`, puis la remplit avec une double boucle. Chaque pixel `(i, j)` de la nouvelle image prend la valeur du pixel `(i × facteur, j × facteur)` de l'image d'origine, c'est-à-dire le pixel en haut à gauche de chaque bloc 4×4. Les 15 autres pixels du bloc sont ignorés.

```python
def sous_echantillonner(image, facteur):
    nh, nw = image.shape[0] // facteur, image.shape[1] // facteur
    nouvelle = np.zeros((nh, nw), dtype=np.uint8)
    for i in range(nh):
        for j in range(nw):
            nouvelle[i, j] = image[i * facteur, j * facteur]
    return nouvelle

img = cv2.imread("peppers-256.png", cv2.IMREAD_GRAYSCALE)
img_boucle = sous_echantillonner(img, 4)
print(img.shape, "->", img_boucle.shape)  # (256, 256) -> (64, 64)
```

On vérifie que l'image obtenue a bien les dimensions attendues : **64×64**.

**Méthode 2 : `cv2.resize` avec `INTER_NEAREST`**

Chaque pixel de l'image réduite prend la valeur du pixel le plus proche dans l'image d'origine. Comme la méthode 1, elle ne garde qu'un pixel par bloc.

**Méthode 3 : `cv2.resize` avec `INTER_AREA`**

Chaque pixel de l'image réduite est calculé à partir de l'ensemble des pixels de la zone correspondante, ici la moyenne du bloc 4×4.

```python
img_nearest = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
img_area    = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
```

| Originale (256×256) | Double boucle (64×64) | `INTER_NEAREST` (64×64) | `INTER_AREA` (64×64) |
|:---:|:---:|:---:|:---:|
| ![](screen/sous-echant_base.png) | ![](screen/sous-echant_1.png) | ![](screen/sous-echant_1.png) | ![](screen/sous-echant_1.png) |

**Pourquoi le passage d'une image 256×256 à une image 64×64 conduit-il à conserver un pixel par bloc de 4×4 ?**

Le facteur de réduction est de 256 / 64 = 4 dans chaque direction. Chaque pixel de l'image réduite correspond donc à une zone de 4×4 = 16 pixels de l'image d'origine. L'image est découpée en 64×64 blocs et on ne conserve qu'une valeur par bloc : le nombre de pixels passe de 65 536 à 4 096, soit une division par 16.

**Comparaison des trois méthodes**

La double boucle et `INTER_NEAREST` donnent le même type de résultat, puisque les deux ne gardent qu'un seul pixel par bloc. On observe des contours crénelés, en marches d'escalier. Les petits détails peuvent disparaître ou être déformés, selon la position du pixel retenu dans chaque bloc. C'est le phénomène d'**aliasing** (repliement spectral) : on réduit l'image sans filtrer au préalable les hautes fréquences, donc les variations rapides d'intensité.

Avec `INTER_AREA`, chaque pixel est la moyenne de son bloc, ce qui agit comme un filtre passe-bas avant la réduction. L'image est plus lisse et plus fidèle à l'originale : les contours sont plus réguliers et les petits détails sont atténués de façon homogène au lieu d'être déformés. En contrepartie, l'image paraît légèrement plus floue. C'est la méthode la plus adaptée à la réduction d'images.

### 3.2 - Sur-échantillonnage

L'objectif est d'agrandir l'image `peppers-128.png` (128×128, niveaux de gris) aux dimensions 512×512, soit un facteur 4 dans chaque direction, et de comparer deux méthodes d'interpolation.

- **`INTER_NEAREST` (plus proche voisin)** : chaque nouveau pixel prend la valeur du pixel source le plus proche.
- **`INTER_LINEAR` (bilinéaire)** : chaque nouveau pixel est une moyenne pondérée des 4 pixels voisins (2×2), selon leur distance.

```python
img128 = cv2.imread("peppers-128.png", cv2.IMREAD_GRAYSCALE)
img_nearest = cv2.resize(img128, (512, 512), interpolation=cv2.INTER_NEAREST)
img_linear  = cv2.resize(img128, (512, 512), interpolation=cv2.INTER_LINEAR)
```

| Plus proche voisin (512×512) | Bilinéaire (512×512) |
|:---:|:---:|
| ![](screen/sur-echant_1.png) | ![](screen/sur-echant_2.png) |

Les deux images sont affichées à la même échelle pour pouvoir comparer les contours et les zones à fortes variations d'intensité.

**Différences entre le plus proche voisin et l'interpolation bilinéaire**

Avec le plus proche voisin, chaque pixel de l'image 128×128 est recopié en un bloc de 4×4 pixels. L'image est fortement pixelisée : les blocs sont visibles et les contours ont un aspect en escalier, particulièrement dans les zones à fortes variations d'intensité, comme les bords des poivrons. En revanche, aucune nouvelle valeur d'intensité n'est créée.

Avec l'interpolation bilinéaire, les valeurs varient progressivement entre les pixels d'origine. Les blocs disparaissent et les contours sont plus lisses, mais l'image paraît floue : les transitions nettes sont étalées sur plusieurs pixels par le moyennage.

**L'agrandissement permet-il de retrouver des détails absents de l'image 128×128 ?**

Non. L'image 128×128 ne contient que 16 384 pixels, et les détails fins (hautes fréquences) qu'elle ne contient pas sont définitivement perdus. Lors de l'agrandissement, les nouveaux pixels sont calculés uniquement à partir des pixels existants, par recopie (plus proche voisin) ou par moyenne pondérée (bilinéaire). Aucune information nouvelle n'est créée. L'image 512×512 contient 16 fois plus de pixels, mais pas plus de détails que l'image 128×128. L'interpolation change seulement l'aspect visuel : des blocs d'un côté, du flou de l'autre.

## 4 - Quantification, LUT et plans de couleur

### 4.1 - Quantification des niveaux de gris

