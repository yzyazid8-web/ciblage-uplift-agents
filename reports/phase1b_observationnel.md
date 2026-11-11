# Phase 1 bis : estimation causale hors essai randomisé

Généré le 2026-11-11 par `python -m ciblage.causal.observational`.

Action étudiée : **email_homme** sur le résultat **visit**. Covariable de confusion injectée : **history** (intensité 1.5).

## 1. Vérité terrain (données randomisées complètes)

Sur les 42 613 clients randomisés (témoin + email_homme), l'effet mesuré est **+7.66 pts** [+7.00 ; +8.32]. C'est la cible à retrouver.

## 2. Le biais de confusion

Après avoir cassé la randomisation, l'échantillon observationnel compte 21 184 clients (9 847 traités, 11 337 témoins). La **différence de moyennes naïve** y donne **+10.44 pts** [+9.49 ; +11.39] : soit un biais de **+2.78 pts** par rapport à la vérité. Le confondant gonfle l'effet apparent.

## 3. Recouvrement et rééquilibrage par la propension

- Propension estimée dans [0.214 ; 0.990], part de clients extrêmes (e<0,05 ou e>0,95) : 5.3 %.
- Effectif efficace conservé : 80 % des traités, 52 % des témoins (recouvrement **limité**).
- **|SMD| maximale** entre groupes : **0.940** sans pondération → **0.094** après pondération par 1/propension (seuil d'équilibre 0,1). La propension rééquilibre bien les groupes.

## 4. Correction du biais

« écart à la vérité » = effet estimé − vérité terrain ; plus il est proche de 0, mieux c'est. L'AIPW est doublement robuste : juste si le modèle de réponse OU la propension l'est.

| méthode | effet estimé (pts) | IC 95 % (pts) | écart à la vérité (pts) | significatif |
|---|---|---|---|---|
| naïf | +10.44 | [+9.49 ; +11.39] | +2.78 | oui |
| IPW | +7.82 | [+6.56 ; +9.07] | +0.16 | oui |
| AIPW | +7.79 | [+6.55 ; +9.03] | +0.13 | oui |

## 5. Et si le confondant n'était pas mesuré ?

Démonstration directe plutôt qu'un indice théorique : on refait l'AIPW en **retirant `history` de l'ajustement** (propension et réponse), comme si ce confondant n'avait pas été observé.

- AIPW **avec** `history` (ajustement complet) : **+7.79 pts** [+6.55 ; +9.03], écart à la vérité +0.13 pts.
- AIPW **sans** `history` (confondant non mesuré) : **+8.96 pts** [+7.96 ; +9.97], écart +1.31 pts.
Retirer le confondant fait **revenir le biais** : aucune méthode ne corrige un confondant qu'elle ne voit pas. La correction ne vaut donc que si les bons déterminants sont mesurés.

## 6. Robustesse selon l'intensité du confounding

On fait varier l'intensité du biais injecté (0 = randomisation préservée). Le biais naïf croît avec l'intensité, tandis que l'AIPW reste proche de la vérité tant que le confondant est mesuré.

| intensité | clients | naïf (pts) | écart naïf | AIPW (pts) | écart AIPW |
|---|---|---|---|---|---|
| 0.0 | 21 123 | +7.78 | +0.12 | +7.72 | +0.06 |
| 0.5 | 21 154 | +8.65 | +0.99 | +7.59 | -0.06 |
| 1.0 | 21 227 | +9.78 | +2.12 | +8.01 | +0.35 |
| 1.5 | 21 184 | +10.44 | +2.78 | +7.79 | +0.13 |
| 2.0 | 21 197 | +10.86 | +3.20 | +8.33 | +0.67 |
