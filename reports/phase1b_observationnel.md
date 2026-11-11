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

## 5. Sensibilité au confounding caché (E-value)

Même corrigé, l'effet pourrait venir d'un confondant **non mesuré**. La E-value est le risque relatif minimal qu'un tel confondant devrait avoir avec le traitement ET le résultat pour annuler l'effet.

- Risque relatif implicite de l'AIPW : **1.85**.
- **E-value ponctuelle : 3.10** — un confondant caché devrait être associé d'un facteur ≥ 3.10 au traitement et au résultat pour expliquer l'effet.
- **E-value de l'IC : 2.82** (l'IC exclut 0 : un confondant plus faible que cela ne suffirait pas à rendre le résultat non significatif).
