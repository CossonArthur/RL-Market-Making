import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from sortedcontainers import SortedDict
from typing import List, Optional, Tuple, Union, Deque, Dict


from scipy.linalg import expm


@dataclass
class Order:
    order_id: str
    place_ts: float
    exchange_ts: float
    side: str
    price: float
    size: float


@dataclass
class CancelOrder:
    exchange_ts: float
    id_to_delete: int


@dataclass
class Trade:
    exchange_ts: float
    receive_ts: float
    side: str
    price: float
    quantity: float


@dataclass
class OrderBook:
    exchange_ts: float
    receive_ts: float
    asks: List[Tuple[float, float]]  # tuple[price, size]
    bids: List[Tuple[float, float]]  # tuple[price, size]


@dataclass
class OwnTrade:
    placing_ts: float
    exchange_ts: float
    receive_ts: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float
    execute: str  # BOOK or TRADE


@dataclass
class MarketEvent:
    exchange_ts: float
    receive_ts: float
    orderbook: Optional[OrderBook] = None
    trade: Optional[Trade] = None


def update_best_positions(
    best_bid, best_ask, market_event: MarketEvent, levels: bool = False
) -> Tuple[float, float]:
    """
    Update best bid and best ask prices

    Args:
        best_bid(float): best bid price
        best_ask(float): best ask price
        market_event(MarketEvent): market data
        levels(bool): return  levels or not

    Returns:
        best_bid(float): best bid price
        best_ask(float): best ask price
        asks_price(List[float]): ask price levels
        bids_price(List[float]): bid price levels
        asks_volume(List[float]): ask volume levels
        bids_volume(List[float]): bid volume levels

    """

    if market_event.trade is None:  # MarketEvent is OrderBook
        best_bid = market_event.orderbook.bids[0][0]
        best_ask = market_event.orderbook.asks[0][0]

        # return ask and bid levels
        if levels:
            asks_price = [level[0] for level in market_event.orderbook.asks]
            bids_price = [level[0] for level in market_event.orderbook.bids]

            asks_volume = [level[0] for level in market_event.orderbook.asks]
            bids_volume = [level[0] for level in market_event.orderbook.bids]

            return best_bid, best_ask, asks_price, bids_price, asks_volume, bids_volume
        else:
            return best_bid, best_ask

    else:  # MarketEvent is Trade
        if market_event.trade.side == "buy":
            best_ask = max(market_event.trade.price, best_ask)
        elif market_event.trade.side == "sell":
            best_bid = min(best_bid, market_event.trade.price)
        return best_bid, best_ask


class Real_Data_Env:
    def __init__(
        self,
        market_data: List[MarketEvent],
        execution_latency: float,
        market_event_latency: float,
    ) -> None:
        """
        Args:
            market_data(List[MarketEvent]): market data
            execution_latency(float): latency in nanoseconds
            market_event_latency(float): latency in nanoseconds
        """

        self.price_history = []

        # market event queue
        self.market_event_queue = deque(market_data)
        # agent action queue
        self.actions_queue: Deque[Union[Order, CancelOrder]] = deque()
        # all the event that will be sent to the strategy : SordetDict: receive_ts -> [updates]
        self.strategy_updates_queue = SortedDict()
        # map : order_id -> Order
        self.ready_to_execute_orders: Dict[int, Order] = {}

        # current market_event
        self.market_event: Optional[MarketEvent] = None

        # current ids
        self.order_id = 0
        self.trade_id = 0
        # current bid and ask
        self.best_bid = -np.inf
        self.best_ask = np.inf
        # current trade
        self.trade_price = {}
        self.trade_price["buy"] = -np.inf
        self.trade_price["sell"] = np.inf
        # last order
        self.last_order: Optional[Order] = None

        # latency
        self.latency = execution_latency
        self.market_event_latency = market_event_latency

    def get_market_event_queue_event_time(self) -> float:
        return (
            np.inf
            if len(self.market_event_queue) == 0
            else self.market_event_queue[0].exchange_ts
        )

    def get_actions_queue_event_time(self) -> float:
        return (
            np.inf
            if len(self.actions_queue) == 0
            else self.actions_queue[0].exchange_ts
        )

    def get_strategy_updates_queue_event_time(self) -> float:
        return (
            np.inf
            if len(self.strategy_updates_queue) == 0
            else self.strategy_updates_queue.keys()[0]
        )

    def get_order_id(self) -> int:
        res = self.order_id
        self.order_id += 1
        return res

    def get_trade_id(self) -> int:
        res = self.trade_id
        self.trade_id += 1
        return res

    def update_best_pos(self) -> None:
        assert not self.market_event is None, "no current market data!"

        self.best_bid = self.market_event.orderbook.bids[0][0]
        self.best_ask = self.market_event.orderbook.asks[0][0]

        self.price_history.append((self.best_bid + self.best_ask) / 2)

    def update_last_trade(self) -> None:
        assert not self.market_event is None, "no current market data!"

        self.trade_price[self.market_event.trade.side] = self.market_event.trade.price

    def delete_last_trade(self) -> None:
        self.trade_price["buy"] = -np.inf
        self.trade_price["sell"] = np.inf

    def update_market_event(self, market_event: MarketEvent) -> None:

        # current orderbook
        self.market_event = market_event

        # update the environment
        if self.market_event.orderbook is None:
            self.update_last_trade()
        else:
            self.update_best_pos()

        # add market_event to strategy_updates_queue
        if not market_event.receive_ts in self.strategy_updates_queue.keys():
            self.strategy_updates_queue[market_event.receive_ts] = []
        self.strategy_updates_queue[market_event.receive_ts].append(market_event)

    def update_action(self, action: Union[Order, CancelOrder]) -> None:

        if isinstance(action, Order):
            # save last order to try to execute it aggressively
            self.last_order = action
        elif isinstance(action, CancelOrder):
            # cancel order
            if action.id_to_delete in self.ready_to_execute_orders:
                self.ready_to_execute_orders.pop(action.id_to_delete)
        else:
            assert False, "Wrong action type!"

    def tick(self) -> Tuple[float, List[Union[OwnTrade, MarketEvent]]]:
        """
        Simulation tick

        Returns:
            receive_ts(float): receive timestamp in nanoseconds
            res(List[Union[OwnTrade, MarketEvent]]): simulation result.
        """
        while True:
            # get event time for all the queues
            strategy_updates_queue_et = self.get_strategy_updates_queue_event_time()
            market_event_queue_et = self.get_market_event_queue_event_time()
            actions_queue_et = self.get_actions_queue_event_time()

            # if both queue are empty : no more events
            if market_event_queue_et == np.inf and actions_queue_et == np.inf:
                break

            # strategy queue has minimum event time : execute it
            if strategy_updates_queue_et < min(market_event_queue_et, actions_queue_et):
                break

            if market_event_queue_et <= actions_queue_et:
                self.update_market_event(self.market_event_queue.popleft())
            if actions_queue_et <= market_event_queue_et:
                self.update_action(self.actions_queue.popleft())

            # execute last order aggressively
            self.execute_last_order()
            # execute orders with current orderbook
            self.execute_orders()
            # delete last trade
            self.delete_last_trade()

        # end of simulation
        if len(self.strategy_updates_queue) == 0:
            return np.inf, None
        key = self.strategy_updates_queue.keys()[0]
        res = self.strategy_updates_queue.pop(key)
        return key, res

    def execute_last_order(self) -> None:
        """
        this function tries to execute self.last order aggressively
        """
        # nothing to execute
        if self.last_order is None:
            return

        executed_price, execute = None, None
        #
        if self.last_order.side == "buy" and self.last_order.price >= self.best_ask:
            executed_price = self.best_ask
            execute = "BOOK"
        #
        elif self.last_order.side == "sell" and self.last_order.price <= self.best_bid:
            executed_price = self.best_bid
            execute = "BOOK"

        if executed_price is not None:
            executed_order = OwnTrade(
                self.last_order.place_ts,  # when we place the order
                self.market_event.exchange_ts,  # exchange ts
                self.market_event.exchange_ts + self.market_event_latency,  # receive ts
                self.get_trade_id(),  # trade id
                self.last_order.order_id,
                self.last_order.side,
                self.last_order.size,
                executed_price,
                execute,
            )

            # add order to strategy update queue
            if not executed_order.receive_ts in self.strategy_updates_queue:
                self.strategy_updates_queue[executed_order.receive_ts] = []
            self.strategy_updates_queue[executed_order.receive_ts].append(
                executed_order
            )
        else:
            self.ready_to_execute_orders[self.last_order.order_id] = self.last_order

        # delete last order
        self.last_order = None

    def execute_orders(self) -> None:
        executed_orders_id = []
        for order_id, order in self.ready_to_execute_orders.items():

            executed_price, execute = None, None

            #
            if order.side == "buy" and order.price >= self.best_ask:
                executed_price = order.price
                execute = "BOOK"
            #
            elif order.side == "sell" and order.price <= self.best_bid:
                executed_price = order.price
                execute = "BOOK"
            #
            elif order.side == "buy" and order.price >= self.trade_price["sell"]:
                executed_price = order.price
                execute = "TRADE"
            #
            elif order.side == "sell" and order.price <= self.trade_price["buy"]:
                executed_price = order.price
                execute = "TRADE"

            if not executed_price is None:
                executed_order = OwnTrade(
                    order.place_ts,  # when we place the order
                    self.market_event.exchange_ts,  # exchange ts
                    self.market_event.exchange_ts
                    + self.market_event_latency,  # receive ts
                    self.get_trade_id(),  # trade id
                    order_id,
                    order.side,
                    order.size,
                    executed_price,
                    execute,
                )

                executed_orders_id.append(order_id)

                # add order to strategy update queue
                if not executed_order.receive_ts in self.strategy_updates_queue:
                    self.strategy_updates_queue[executed_order.receive_ts] = []
                self.strategy_updates_queue[executed_order.receive_ts].append(
                    executed_order
                )

        # deleting executed orders
        for k in executed_orders_id:
            self.ready_to_execute_orders.pop(k)

    def place_order(self, ts: float, size: float, side: str, price: float) -> Order:
        order = Order(self.get_order_id(), ts, ts + self.latency, side, price, size)
        self.actions_queue.append(order)
        return order

    def cancel_order(self, ts: float, id_to_delete: int) -> CancelOrder:
        ts += self.latency
        delete_order = CancelOrder(ts, id_to_delete)
        self.actions_queue.append(delete_order)
        return delete_order


class Sim_Env:
    """
    Sim_Env is an object that is able to simulate the order book and its filling with probabilistic orders.

    Parameters
    ----------
    T : integer
        the number of time steps the model will be run for
    dt : integer
        the size of the time steps
    Q : integer
        the maximum (absolute) allowed held volume
    dq : integer
        volume increments
    Q_0 : integer
        the starting volume of the market maker
    dp : float
        the tick size
    min_dp : integer
        the minimum number of ticks from the mid price the market maker has to put their ask/bid price
    mu : float
        the starting price
    std : float
        the standard deviation of the price movement
    lambda_pos : float
        the intensity of the Poisson process dictating the arrivals of sell MOs
    lambda_neg : float
        the intensity of the Poisson process dictating the arrivals of buy MOs
    kappa : float
        the parameter of the execution probability of MOs
    alpha : float
        fee for taking liquidity (posting MOs)
    phi : float
        the running inventory penalty parameter
    pre_run : integer
        the number of time steps the price process should be run before simulating
    printing : bool
        whether or not information should be printed during simulation
    debug : bool
        whether or not information for debugging should be printed during simulation
    d : int
        number of ticks away from mid price the market maker can quote
    use_all_times : bool
        boolean indicating whether to use all time steps or an indicator
    analytical : bool
        whether or not analytical depths should be used
    breaching_penalty : bool
        whether or not a breaching_penalty should be used
    breach_penalty : float
        the penalty for breaching the inventory limit
    reward_scale : float
        a scaling factor used for the reward
    breach_penalty_function : function
        function used to transform the exceeded inventory to a penalty
    """

    def __init__(
        self,
        T=10,
        dt=1,
        Q=3,
        dq=1,
        Q_0=0,
        dp=0.1,
        min_dp=1,
        mu=100,
        std=0.01,
        lambda_pos=1,
        lambda_neg=1,
        kappa=100,
        alpha=1e-4,
        phi=1e-5,
        pre_run=None,
        printing=False,
        debug=False,
        d=5,
        use_all_times=True,
        analytical=False,
        breaching_penalty=False,
        breach_penalty=20,
        reward_scale=1,
        breach_penalty_function=np.square,
    ):

        self.T = T  # maximal time
        self.dt = dt  # time increments

        self.Q = Q  # maximal volume
        self.dq = dq  # volume increments the agent can work with
        self.Q_0 = Q_0  # the starting volume

        self.dp = dp  # tick size
        self.min_dp = min_dp  # the minimum number of ticks from the mid price the market maker has to put their ask/bid price
        self.d = d  # Number of ticks away the market maker is allowed to quote

        self.mu = mu  # average price of the stock
        self.std = std  # standard deviation the price movement

        self.alpha = alpha  # penalty for holding volume
        self.phi = phi  # the running inventory penalty parameter

        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.kappa = kappa
        self.mm_bid = None
        self.mm_ask = None

        self.z = None
        self.A = None
        self.init_analytically_optimal()

        self.printing = printing
        self.debug = debug

        self.analytical = analytical
        self.breach_penalty = breach_penalty
        self.breach = False
        self.breaching_penalty = breaching_penalty
        self.breach_penalty_function = breach_penalty_function

        self.reward_scale = reward_scale

        # Reset the environment
        self.reset()

        # Pre-running the price for pre_run time steps
        if pre_run != None:
            self.pre_run(pre_run)  # Ha kvar? Troligtvis inte

        # Remembering the start price for the reward
        self.start_mid = self.mid

    def pre_run(self, n_steps=100):
        """
        Updates the price n_steps times

        Parameters
        ----------
        n_steps : integer
            the number of time steps the price process should be run for

        Returns
        -------
        None
        """

        for _ in range(n_steps):
            self.update_price()

    def update_price(self):
        """
        Updates the mid price once and makes sure it's within bounds

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # The change rounded to the closest tick size
        self.mid += self.round_to_tick(np.random.normal(0, self.std))

    def init_analytically_optimal(self):
        """
        Calculates z and A which will be used for the optimal solution

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.z = np.exp(
            -self.alpha
            * self.kappa
            * np.square(np.array(range(self.Q, -self.Q - 1, -1)))
        )
        self.A = np.zeros((self.Q * 2 + 1, self.Q * 2 + 1))
        for i in range(-self.Q, self.Q + 1):
            for q in range(-self.Q, self.Q + 1):
                if i == q:
                    self.A[i + self.Q, q + self.Q] = -self.phi * self.kappa * (q**2)
                elif i == q - 1:
                    self.A[i + self.Q, q + self.Q] = self.lambda_pos * np.exp(-1)
                elif i == q + 1:
                    self.A[i + self.Q, q + self.Q] = self.lambda_neg * np.exp(-1)

    def calc_analytically_optimal(self):
        """
        Calculates the analytically optimal bid/ask depth for the current time step

        Parameters
        ----------
        None

        Returns
        -------
        action : np array
            np array with bid/ask depth
        """

        omega = np.matmul(expm(self.A * (self.T - self.t)), self.z)
        h = 1 / self.kappa * np.log(omega)

        if self.Q_t != -self.Q:
            delta_pos = 1 / self.kappa - h[self.Q_t + self.Q - 1] + h[self.Q_t + self.Q]
        if self.Q_t != self.Q:
            delta_neg = 1 / self.kappa - h[self.Q_t + self.Q + 1] + h[self.Q_t + self.Q]

        if self.Q_t == -self.Q:
            d_ask = np.Inf
            d_bid = delta_neg
        elif self.Q_t == self.Q:
            d_ask = delta_pos
            d_bid = np.Inf
        else:
            d_ask = delta_pos
            d_bid = delta_neg

        action = np.array([d_bid, d_ask])

        return action

    def discrete_analytically_optimal(self):
        """
        Calculates the analytically optimal bid/ask depth for the current time step in #ticks

        Parameters
        ----------
        None

        Returns
        -------
        action : np array
            np array with the number of ticks away from the mid price
        """

        action = np.rint(self.calc_analytically_optimal() / self.dp) - self.min_dp

        return action

    def transform_action(self, action):
        """
        Transforms an action in number of ticks to the actual difference in bid/ask from the mid price.
        Also includes the minimum distance to the mid price and transforms ask to negative.

        Parameters
        ----------
        action : np array
            the number of ticks away from the tick price for the ask and bid price

        Returns
        -------
        action : np array
            how far away the ask/bid is chosen from the mid
        """

        return (action + self.min_dp) * np.array([-1, 1]) * self.dp

    def step(self, action):
        """
        Takes a step in the environment based on the market maker taking an action

        Parameters
        ----------
        action : np array
            the number of ticks away from the tick price for the ask and bid price

        Returns
        -------
        obs : tuple
            the observation for time step t
        reward : float
            the reward for time step t
        """

        self.t += self.dt

        # ----- UPDATING THE PRICE -----
        self.update_price()

        # Update bid and ask to the given number of ticks away from the mid price
        if self.analytical:
            [self.mm_bid, self.mm_ask] = self.mid + np.array([-1, 1]) * action
        else:
            [self.mm_bid, self.mm_ask] = self.mid + self.transform_action(action)

        if self.debug:
            print("Starting volume for time step:", self.Q_t)
            print("The mid price is:", self.mid)
            print("The action is:", action)
            print("The choice is:", self.mm_bid, "|", self.mm_ask)

        # ----- TAKING THE ACTION -----
        # In this case the MM always updates the bid and ask to two ticks away from the mid with volume = 1

        # If we're not at the final time step the MM can do what it wants
        if self.t <= self.T:
            # ----- SAMPLE ORDERS -----

            # The number of orders that arrive
            n_MO_buy = np.random.poisson(self.lambda_neg)
            n_MO_sell = np.random.poisson(self.lambda_pos)

            # The probability that the orders get executed
            p_MO_buy = np.exp(-self.kappa * (self.mm_ask - self.mid))
            p_MO_sell = np.exp(-self.kappa * (self.mid - self.mm_bid))

            # Sample the number of orders executed
            n_exec_MO_buy = np.random.binomial(n_MO_buy, p_MO_buy)
            n_exec_MO_sell = np.random.binomial(n_MO_sell, p_MO_sell)

            # Step 1: add cash from the arriving buy and sell MOs that perfectly cancel each other
            if n_exec_MO_buy * n_exec_MO_sell > 0:
                self.X_t += (self.mm_ask - self.mm_bid) * np.min(
                    [n_exec_MO_buy, n_exec_MO_sell]
                )

            # Note to self: there shouldn't be any issues with infinite bids and asks since the probability of a
            # filled LO at +/- inf is zero

            # Step 2: compute the net balance from time step t for the market maker, and make adjustments
            # Net balance for the market maker is the difference in arriving MO sell and arriving MO buy orders
            n_MO_net = n_exec_MO_sell - n_exec_MO_buy

            if (
                n_MO_net != 0
            ):  # Saving computational power, if net above is zero, skip below
                # Determine if net balance would result in a limit breach, and if so, adjust accordingly to keep
                # within limits
                if n_MO_net + self.Q_t > self.Q:  # long inventory limit breach
                    self.breach = self.breach_penalty_function(
                        n_MO_net + self.Q_t - self.Q
                    )
                    n_MO_net = (
                        self.Q - self.Q_t
                    )  # the maximum allowed net increase is given by Q - Q_t
                    n_exec_MO_sell -= n_MO_net + self.Q_t - self.Q
                elif n_MO_net + self.Q_t < -self.Q:  # short inventory limit breach
                    self.breach = self.breach_penalty_function(
                        -self.Q - (n_MO_net + self.Q_t)
                    )
                    n_MO_net = (
                        -self.Q - self.Q_t
                    )  # the maximum allowed net decrease is given by -Q + Q_t
                    n_exec_MO_buy -= -self.Q - (n_MO_net + self.Q_t)
                else:
                    self.breach = False

                # Step 3: add cash from net trading
                if n_MO_net > 0:
                    self.X_t -= self.mm_bid * n_MO_net
                elif n_MO_net < 0:
                    self.X_t -= self.mm_ask * n_MO_net

                self.Q_t += n_MO_net

            if self.debug:
                print("Arrvials:")
                print(n_MO_buy)
                print(n_MO_sell)

                print("\nProbabilities:")
                print(p_MO_buy)
                print(p_MO_sell)

                print("\nExecutions:")
                print(n_exec_MO_buy)
                print(n_exec_MO_sell)
                print("Net:", n_MO_net)
                print("Net:", n_exec_MO_sell - n_exec_MO_buy)

                print("\nX_t:", self.X_t)
                print("Q_t:", self.Q_t)

                print("_" * 20)

        # The time is up!
        if self.t == self.T:
            # The MM liquidates their position at a worse price than the mid price

            # The MM has to buy at their ask and sell at their bid with an additional unfavorable discount/increase
            self.X_t += self.final_liquidation()

            self.Q_t = 0

        # ----- THE REWARD -----
        V_t_new = self.X_t + self.H_t()
        if self.t <= self.T:
            reward = self._get_reward(V_t_new)
        else:
            reward = 0

        # ----- UPDATE VALUES -----
        self.V_t = V_t_new  # the cash value + the held value

        # ----- USEFUL INFORMATION -----
        if self.printing:
            print("The reward is:", reward)
            self.render()

        return self.state(), reward

    def _get_reward(self, V_t):
        """
        Returns the reward for the current time step

        Parameters
        ----------
        V_t : float
            the value process for the previous time step

        Returns
        -------
        reward : float
            the reward for time step t
        """

        # New value minus old
        # Subtract penalty for held volume

        if self.breaching_penalty:
            return (
                self.reward_scale * (V_t - self.V_t)
                + self.inventory_penalty()
                - self.breach_penalty * self.breach
            )
        else:
            return self.reward_scale * (V_t - self.V_t) + self.inventory_penalty()

    def reset(self):
        """
        Resets the environment

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.mid = self.mu
        self.Q_t = self.Q_0
        self.V_t = self.H_t()  # the value process involves no cash at the start
        self.t = 0
        self.X_t = 0  # the cash process
