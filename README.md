# Bengali.AI Handwritten Grapheme

## Introduction
Bengali is the 5th most spoken language in the world with hundreds of million of speakers. Optical character recognition is particularly challenging for Bengali. While Bengali has 49 letters (to be more specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This means that there are many more graphemes, or the smallest units in a written language. The added complexity results in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).

Bangladesh-based non-profit Bengali.AI is focused on helping to solve this problem. They build and release crowdsourced, metadata-rich datasets and open source them through research competitions. Through this work, Bengali.AI hopes to democratize and accelerate research in Bengali language technologies and to promote machine learning education.

For this competition, we are given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.


## Utilisation du code

Création d'un environnement virtuel

Pour utiliser le code de ce dépôt, il faut d'abord créer un environnement virtuel avec virtualenv :

Se placer à la racine du projet
Installer virtualenv via pip : pip install virtualenv
Créer l'environnement virtuel : virtualenv .env
activer l'envirennoment virtuel : source .env/bin/activate
installer les librairies : pip install -r requirements.txt
Lancement de l'entraînement du modèle

##Pour faire tourner le script principal main.py :

python main.py : permet d'entrainer le modèle et d'obtenir une évaluation des performances sur le jeu d'entrainement