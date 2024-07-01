# Market Making Algorithm with RL Techniques

This repository contains an implementation of a **market making** algorithm that utilizes reinforcement learning (**RL**) techniques with **inventory constrains**.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Methods](#methods)
  - [State space](#state-space)
  - [Action space](#action-space)
  - [Reward](#reward)
- [Results](#results)
- [Usage](#usage)
- [Repository Structure](#repository-structure)

## Introduction

In the field of finance, market making refers to the process of providing liquidity to financial markets by continuously quoting bid and ask prices for a particular asset. Market making algorithms aim to optimize the bid-ask spread and manage inventory levels to maximize profitability.


## Data
The data used for this repository is the [Crypto Lake website](https://crypto-lake.com/). The data is **High Frequency** and contains the order book and the trades data for the "**BTC/USDT**" trading pair as well as the trades on the Binance exchange. It contains 20 price levels for each order book side for each timestamp(~850k) of the day "1/10/2022" and the trades that occured on that day (~2.5M). The data is stored in a queue and the last market event is fetch to simulate a real-time market data feed.

## Methods

### State space 
The state space is composed of the features study in the data preparation part. It is composed of the following features:
- Inventory ratio
$$ IR_t = \frac{Q_t - Q_{\text{min}}}{Q_{\text{max}} + Q_{\text{min}}}$$
- Books imbalance
$$\text{Book Imbalance} = \frac{\text{Total Bids} - \text{Total Asks}}{\text{Total Bids} + \text{Total Asks}}$$
- Volatility
$$\text{Volatility} = std(\frac{m_t - m_{t-1}}{m_{t-1}})$$
- RSI (Relative Strength Index)
$$\text{RSI} = 100 - \frac{100}{1 + \frac{\text{EMA Gain}}{\text{EMA Loss}}}$$

Each features and discretized into 10 bins. 



### Action space
The action space is **discrete** and is a tuple of two elements: 
- the **depth of the asks**
- the **depth of the bids**

*For instance, (1,3) means that the agent will place a sell order at the first bid price and a buy order at the third ask price.*


### Reward
The rewards *awards* the agent when he is **making profit** from either the **spread** or the **inventory**. He is *penalized* when he is **holding an order for too long** and when **the risk** associated to being exposed to variations in the mid price is too high, as well as when the agent holds more than the maximum inventory.
The reward is calculated as using (Sadighian, J. (2020). Extending deep reinforcement learning frameworks in cryptocurrency market making)
$$
    r_t = \left\{
    \begin{array}{ll}
        x_t^{\text{size}} * \delta_t - 10^3  |\frac{Q_t - Q_{\text{min}}}{Q_{\text{max}} - Q_{\text{min}}} - 0.5|^2 & \text{order executed} \\
        - \frac{\Delta t_{\text{hold}}}{\Delta t_{\text{order}}} & \text{order not executed} \\
        - 10^2 & \text{inventory above borders}
    \end{array}
\right.
$$

where $Q_t$ is the inventory at time t,

$\delta_t$ is the spread at time t,

$m_t$ is the mid price at time t, 

$x_t^{\text{side}}$ is the side of the order at time t, 

$x_t^{\text{size}}$ is the size of the order at time t, 

$x_t^{\text{price}}$ is the price of the order at time t

$\Delta t_{\text{hold}}$ is the time the agent has held the order

$\Delta t_{\text{order}}$ is minimum time interval in between orders

## Results

### Q Learning

### SARSA

### Naive Strategy

### Stoikov Strategy



## Installation

To use this market making algorithm, follow these steps:

1. Clone the repository
2. Install the required dependencies: 
```bash
poetry install
```

## Usage

analysis.ipynb : This notebook contains the analysis of the data

market_making.ipynb : This notebook contains the implementation of the market making algorithm

## Repository Structure
/enviroment : Contains the enviroment for the market making algorithm

/strategies : Contains the strategies for the market making algorithm

/utils : Contains the utility functions used in all the notebooks

/ models : Contains the models used in the market making algorithm