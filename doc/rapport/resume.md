# Rapport: liste des chapitres	

Consignes

https://docs.google.com/document/d/1HytWVtQqaVrpeAMHay2dsBTXUWB_GTnW1SdYKMtnhoY/edit

## Résumé (abstract)



## Table des matières

## Table des figures

## Acronymes, Conventions

## Introduction

#### Contexte: 

- Machine learning et apprentissage par renforcement

#### Objectif

- Apprendre le reinforcement learning
- Apprendre un framework dédié 
- Faciliter l'apprentissage en découpant l'objectif en diverse tâches. L'agent apprend d'abord la première tâche, puis la deuxième, puis il apprend à jongler entre les 2. 

#### Réalisation

- Présentation de l'environnement (World.py)
- Résumer le fonctionnent du code (en très général)

#### Définitions 

- Agent, apprentissage par renforcement? etc



## Théorie

#### Notions de base 

- Fonctionnement général de l'apprentissage
- Valeur d'un état
  - Discount factor
  - equation de bellman

#### Q-Learning

- Valeur d'état / action Q(s,a)

#### Réseaux neuronaux

#### Deep Q-learning

- [...]


- Explore vs Exploit



## Conception

#### Architecture

#### Fonctionnement

#### Points-Clefs



## Résultats

#### Difficultés

#### Reward function

- Son importance et effets. Noté que l'agent faisat toujours la même chose avec une mauvaise reward function. 

#### Batch size et Copy Target period

- Ce n'est qu'à ce moment là que l'apprentissage a réellement commencé

#### Choix du learning rate

#### Paramètres finaux utilisés pour l'apprentissage



## Annexes

- Installation
- Utilisation
  - World
    - Configuration
  - Voir l'agent en action
- Code