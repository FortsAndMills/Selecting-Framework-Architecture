## Selecting Framework Architecture

It is time to make some wise design choices...

Option 1) Everything is one object \
Option 2) Standard modular structure \
Option 3) [Yanush Version](https://codeshare.io/2E4boP)

# Option 2:

## DQN (no target)

![](https://github.com/FortsAndMills/Selecting-Framework-Architecture/blob/master/Design%20Choice/DQN%20(no%20target).png)

## DQN

![](https://github.com/FortsAndMills/Selecting-Framework-Architecture/blob/master/Design%20Choice/DQN.png)

**НЕДОСТАТКИ**: 
* очень громоздкая конструкция ради всего лишь таргет-сетки. Проблема в том, что это придётся прописывать в инициализации.
* блок трейнера, разветвляюдего на два трейнера. Можно чтобы раннер запускал сразу несколько тренеров, но пользователь (!) должен убедиться, что они "независимые" - это плохо.

## Rainbow

![](https://github.com/FortsAndMills/Selecting-Framework-Architecture/blob/master/Design%20Choice/Rainbow.png)

**НЕДОСТАТКИ**: 
* тройное наследование Noisy Dueling Categorical. Но это неизлечимая проблема, не связанная с выбором архитектуры... Здесь скорее всего так или иначе появятся фабрики классов.

## Twin DQN (shared backbone)

![](https://github.com/FortsAndMills/Selecting-Framework-Architecture/blob/master/Design%20Choice/Twin%20DQN%20(shared%20backbone).png)

## Twin DQN

![](https://github.com/FortsAndMills/Selecting-Framework-Architecture/blob/master/Design%20Choice/Twin%20DQN.png)

**ПРОБЛЕМА**: 
* в реплей буффер внезапно попадает сразу две копии каждого трашнзишна!..

## A2C

![](https://github.com/FortsAndMills/Selecting-Framework-Architecture/blob/master/Design%20Choice/A2C.png)

**ПРОБЛЕМА**: 
* лоссы не обмениваются информацией. То есть Policy Gradient Loss придётся заново прогонять сеть-критика, а энтропийному лоссу - заново прогонять политику.

## QAC

![](https://github.com/FortsAndMills/Selecting-Framework-Architecture/blob/master/Design%20Choice/QAC.png)

**ПРОБЛЕМА**: 
* внезапно требует ОСОБОГО трейнера, работающего сразу с двумя буфферами. Как приоритизрованный реплей тогда сюда пихать?

## Curiosity with A2C

![](https://github.com/FortsAndMills/Selecting-Framework-Architecture/blob/master/Design%20Choice/Curiosity.png)

**НЕДОСТАТКИ**:
* повторное сэмплирование батча из реплей буффера для обновления сетки для любопытства.
* и наследование всех проблем от предыдущих пунктов.
