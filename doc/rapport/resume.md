# Rapport: liste des chapitres	

Consignes

https://docs.google.com/document/d/1HytWVtQqaVrpeAMHay2dsBTXUWB_GTnW1SdYKMtnhoY/edit

## Résumé (abstract)



## Table des matières

## Table des figures

## Acronymes, Conventions

Mettre tous les hyper-paramètres ici

## Introduction

#### Contexte: 

- Machine learning et apprentissage par renforcement.
- Indiquer que ce sujet n'est pas enseigné dans le cadre du bachelor, et que c'est sujet à bcp de recherches et les techniques évoluent sans cesse.  La théorie qui va suivre date d'il y a 2/3 ans, et bcp de techniques plus performantes ont vu le jour. Mais la base reste la même dans le cadre du réinforcement learning. 

#### Objectif

- Apprendre le reinforcement learning
- Apprendre un framework dédié 
- Faciliter l'apprentissage en découpant l'objectif en diverse tâches. L'agent apprend d'abord la première tâche, puis la deuxième, puis il apprend à jongler entre les 2. 

#### Réalisation

- Présentation de l'environnement (World.py)
- Résumer le fonctionnent du code (en très général)

#### Définitions --> ds théorie? 

- Agent, apprentissage par renforcement? etc



## Théorie

#### Notions de base 

- Fonctionnement général de l'apprentissage
- Valeur d'un état
  - Discount factor
  - equation de bellman

#### Q-Learning

- explore vs exploit


- Valeur d'état / action Q(s,a)
- présentation algorithme Q-Learning
- Limitations Q-Learning

#### Réseaux neuronaux

#### Deep Q-learning

- [...]


- Explore vs Exploit


## Conception

- expliquer quelque part le où se trouve les codes implémentant le pseudo-code vu dans la partie DQN
- parler du fait que l'on utiliser la librairie tensorflow afin de s'épargner tous les calculs de prédiction et backpropagation

#### Architecture

- présenter vite fait l'interface de World. 

#### Fonctionnement

#### Points-Clefs



## Résultats

#### Difficultés

#### Reward function

- Son importance et effets. Noté que l'agent faisat toujours la même chose avec une mauvaise reward function. 
- Répéter pourquoi il est plus simple de fractionner l'apprentissage. 

#### Batch size et Copy Target period

- Ce n'est qu'à ce moment là que l'apprentissage a réellement commencé

#### Choix du learning rate

#### Paramètres finaux utilisés pour l'apprentissage

## Conclusion

- Dire que les apprentissages n'étaient pas forcéements adaptés. Mais qu'on peut réutiliser ceci dans un cadre plus adapté du genre: apprendre à marcher. Si on tombe, appredre à se relever, puis apprendre à marcher. 
- Suite à ce projet, que faire si j'étais encore dessus?

## Annexes

- Installation
- Utilisation
  - World
    - Configuration
  - Voir l'agent en action
- Code