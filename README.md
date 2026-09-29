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

### 3.1 - Sous-échantillonnage*

## 4 - Quantification, LUT et plans de couleur

### 4.1 - Quantification des niveaux de gris

