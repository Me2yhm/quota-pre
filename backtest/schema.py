import math
from datetime import datetime, timedelta
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import TypedDict


import pandas as pd

accuracy = 6


def lcm_float(x, y) -> float:
    multiplier = 10 ** max(len(str(x).split(".")[-1]), len(str(y).split(".")[-1]))
    x_int = int(x * multiplier)
    y_int = int(y * multiplier)
    lcm_int = x_int * y_int // math.gcd(x_int, y_int)
    return round(lcm_int / multiplier, accuracy)


def float_division(x, y) -> float:
    multiplier = 10 ** max(len(str(x).split(".")[1]), len(str(y).split(".")[1]))
    x_int = int(x * multiplier)
    y_int = int(y * multiplier)
    result_int = x_int // y_int
    result_float = result_int / multiplier
    return result_float


class pool(TypedDict):
    breed: str
    volume: int
    price: float
    unit: int


class account(ABC):
    pools: dict[str:pool]
    portfolio_value: float

    def __init__(
        self, base_currency: float, pools: dict[str, pool] = defaultdict(pool)
    ) -> None:
        self.pools = pools
        cash: pool = {
            "breed": "base_currency",
            "volume": base_currency,
            "price": 1,
            "unit": 0,
        }
        self.pools["base_currency"] = cash
        self.cal_portfolio_value()

    def count_volume(self, breed: str) -> int | float:
        """
        count the position of certain breed
        """
        try:
            volume = self.pools[breed]["volume"]
            return volume
        except KeyError:
            print(f"Warning: the breed {breed} not in pools")
            return 0

    def count_base_current(self) -> float:
        """
        count the cash
        """
        return self.count_volume("base_currency")

    def check_cash(self, amount: float = 0) -> float:
        """
        check the cash wether is enough
        """
        cash = self.count_base_current()
        if cash >= amount and cash > 0:
            return amount
        print(f"Warning: Current cash is less than need amount {amount}, get {cash}")
        return cash

    def check_breed(self, breed: str, amount: float | int = 0) -> float:
        volume = self.count_volume(breed)
        if volume >= amount and volume > 0:
            return amount
        print(
            f"Warning: Current {breed} is less than need amount {amount}, get {volume}"
        )
        return volume

    def withdraw_breed(self, breed: str, amount: float | int):
        """
        withdraw breed from pools
        """
        try:
            amount = self.check_breed(breed, amount)
            self.pools[breed]["volume"] -= amount
            if self.pools[breed]["volume"] == 0:
                del self.pools[breed]
            self.cal_portfolio_value()
        except KeyError:
            pass

    def save_breed(
        self,
        breed: str,
        amount: float,
        price: float,
        unit: int = 1,
    ):
        """
        save breed in pools
        """
        try:
            self.pools[breed]["volume"] += amount
            self.update_price(breed, price)
        except KeyError:
            position: pool = {
                "breed": breed,
                "price": price,
                "unit": unit,
                "volume": amount,
            }
            self.pools[breed] = position
        finally:
            self.cal_portfolio_value()

    def handle_cash(self, amount: float):
        """
        if amount >=0, withdraw cash, else save.
        """
        if amount >= 0:
            self.withdraw_breed("base_currency", amount)
        else:
            self.save_breed("base_currency", amount, 1, 0)

    def can_buy_volume(
        self, price_A: float, unit_A, volume_A: int, price_B: float, unit_B: int = 1
    ):
        """
        calculate the volume can buy
        some security has minimum unit to buy, so can-buy volume need more than unit
        unit == 0 meas doesn't have unit
        """
        assert unit_B >= 0, f"unit need not less then 0, but got {unit_B}"
        money_buy = price_A * volume_A
        if unit_B == 0:
            if unit_A != 0:
                assert isinstance(
                    volume_A, int
                ), f"volume A need be int type, but got {volume_A}"
            volume = round(money_buy / price_B, accuracy)
            return volume
        else:
            if unit_A == 0:
                volume = round(price_A * volume_A / price_B / unit_B, accuracy)
                return int(volume)
            assert isinstance(volume_A, int)
            minimum_money = lcm_float(price_A * unit_A, price_B * unit_B)
            if money_buy < minimum_money:
                return 0
            minimum_volume = int(round(minimum_money / price_A, accuracy))
            buy_volume = volume_A - volume_A % minimum_volume
            buy_money = buy_volume * price_A
            volume = round(buy_money / price_B, accuracy)
            return int(volume)

    def trade(
        self, breed_A: str, breed_B: str, volume_A: int, price_B: float, unit: int = 1
    ):
        """
        用品种A去购买品种B
        @parameters:
        breed_A: 品种A
        breed_B: 品种B
        volume_A: 用于支付的品种A的数量
        price_B: 品种B的价格(针对于基础货币而言,也即cash)
        unit: 是否只能整数买(最小购买单位)
        """
        try:
            price_A = self.get_breed_price(breed_A)
            unit_A = self.pools[breed_A]["unit"]
            volume_A = self.check_breed(breed_A, volume_A)
            volume_B = self.can_buy_volume(price_A, unit_A, volume_A, price_B, unit)
            if volume_B < unit:
                return
            volume_A = self.can_buy_volume(price_B, unit, volume_B, price_A, unit_A)
            self.withdraw_breed(breed_A, volume_A)
            self.save_breed(breed_B, volume_B, price_B, unit)
        except KeyError:
            return

    def get_breed_price(self, breed: str) -> float:
        try:
            price = self.pools[breed]["price"]
            return price
        except KeyError:
            print(f"Warning: {breed} not in {self}.pools")
            raise KeyError

    def update_price(self, breed: str, price: float):
        """
        update price of a certain breed if price change
        """
        try:
            self.pools[breed]["price"] = price
            self.cal_portfolio_value()
        except KeyError:
            pass

    def get_pools_df(self) -> pd.DataFrame:
        if self.pools:
            breeds = list(self.pools.keys())
            data = {}
            for col in ["breed", "volume", "price", "unit"]:
                data[col] = [self.pools[bre][col] for bre in breeds]
            return pd.DataFrame(data)

    @abstractmethod
    def buy(self):
        """
        buy security
        """

    @abstractmethod
    def buy_to(self):
        """
        use certain rate of cash to buy securities, regard as cash
        """

    @abstractmethod
    def sell(self):
        """
        sell security
        """

    @abstractmethod
    def sell_to(self):
        """
        sell a certain of securities(need position)
        """

    @abstractmethod
    def order(self):
        """
        generate trade order
        """

    @abstractmethod
    def order_to(self):
        """
        generate trade orders as rate
        """

    @abstractmethod
    def cal_portfolio_value(self):
        """
        calculate portfolio value
        """


class unFeeAaccount(account):
    """
    can not short selling and without transaction fee
    Due to the characteristics of Python floating-point operations, the control accuracy is 6 decimal places
    """

    def buy(self, breed: str, volume: int, price: float, unit: int = 1):
        assert volume >= 0
        need_money = volume * price
        self.trade("base_currency", breed, need_money, price, unit)

    def buy_to(self, breed: str, rate: float, price: float, unit=1):
        assert rate > 0
        cash = self.pools["base_currency"]["volume"]
        need_money = cash * rate
        self.trade("base_currency", breed, need_money, price, unit)

    def sell(self, breed: str, volume: int, price: float, unit=1):
        assert volume >= 0
        self.update_price(breed, price)
        self.trade(breed, "base_currency", volume, 1, unit)

    def sell_to(self, breed: str, rate: float, price: float, unit=1):
        assert rate >= 0
        try:
            assert breed in self.pools.keys()
            self.update_price(breed, price)
            rate = round(1 - rate, accuracy)
            volume = int(self.pools[breed]["volume"] * rate)
            self.sell(breed, volume, price, unit)
        except AssertionError:
            print(f"Didn't have {breed} in pools, can't sell")

    def order(self, breed: str, volume: int, price: float, unit=1):
        if volume >= 0:
            self.buy(breed, volume, price, unit)
        else:
            self.sell(breed, -volume, price, unit)

    def order_to(self, breed: str, rate: int, price: float, unit=1):
        try:
            assert -1 <= rate and rate <= 1
            if rate > 0:
                self.buy_to(breed, rate, price, unit)
            else:
                self.sell_to(breed, -rate, price, unit)
        except AssertionError:
            print("Warning: order rate must belong to [0,1]")

    def cal_portfolio_value(self):
        porfolio_value = 0
        for poo in self.pools.values():
            porfolio_value += poo["volume"] * poo["price"]
        self.portfolio_value = porfolio_value
        return porfolio_value


class futureAccount:
    base: float
    pool: dict[str:int]
    fu_overtoday_fee: float
    fu_intoday_fee: float
    timeline: str
    portfolio_value: float
    benchmark: str
    factors: list[str,]
    cash: float
    stand: str
    transactions: dict[str, list[dict]]
    current_date: str
    margin: float

    def __init__(
        self,
        current_date: str = "20220913",
        base: float = 10000000,
        pool: dict[str:dict] = {},
        fu_overtoday_fee: float = 0.000024,
        fu_intoday_fee: float = 0.00035,
        long_margin: float = 0.12,
        short_margin: float = 0.12,
    ):
        self.current_date = current_date
        self.base = base
        self.cash = base
        self.pool = pool
        self.fate_cash = 0.0
        self.fu_overtoday_fee = fu_overtoday_fee
        self.fu_intoday_fee = fu_intoday_fee
        self.long_margin = long_margin
        self.short_margin = short_margin
        self.margin = 0.0
        self.portfolio_value = self.calculate_portfolio_value()
        self.transactions = defaultdict(list)

    def update_date(self, num: int):
        date_format = "%Y%m%d"
        old_date = datetime.strptime(self.current_date, date_format)
        new_date = (old_date + timedelta(days=num)).strftime(date_format)
        self.current_date = new_date

    def order(self, symbol: str, volumes: int, price: float):
        "按照量买卖证券,限制买空不限制卖空"
        try:
            if symbol in ["IF.CFX", "IH.CFX"]:
                unit = price * 300
            elif symbol in ["IC.CFX", "IM.CFX"]:
                unit = price * 200
            if volumes > 0:
                if self.cash < unit * (self.long_margin + self.fu_overtoday_fee):
                    return
                elif (
                    unit * volumes * (self.long_margin + self.fu_overtoday_fee)
                    > self.cash
                ):
                    volumes = int(
                        self.cash / unit / (self.long_margin + self.fu_overtoday_fee)
                    )
            else:
                if self.cash < unit * (self.short_margin + self.fu_overtoday_fee):
                    return
                elif (
                    unit * (-volumes) * (self.short_margin + self.fu_overtoday_fee)
                    > self.cash
                ):
                    volumes = -int(
                        self.cash / unit / (self.short_margin + self.fu_overtoday_fee)
                    )
            self.fate_cash -= volumes * unit
            self.cash -= abs(volumes * unit) * self.fu_overtoday_fee
            if symbol not in self.pool:
                self.pool[symbol] = {
                    "code": symbol,
                    "volume": volumes,
                    "price": price,
                }
            else:
                self.pool[symbol]["volume"] += volumes
                self.pool[symbol]["price"] = price
            if self.pool[symbol]["volume"] == 0:
                del self.pool[symbol]

            self.calculate_portfolio_value()
            self.transactions[self.current_date].append(
                {
                    "code": symbol,
                    "volume": volumes,
                    "price": price,
                }
            )
        except Exception as e:
            print(e)

    def order_to(self, symbol: str, volumes: float, price: float):
        """购买股票到满仓的指定比例，volumes需要大于零"""
        try:
            if symbol in ["IF.CFX", "IH.CFX"]:
                unit = price * 300
            elif symbol in ["IC.CFX", "IM.CFX"]:
                unit = price * 200
            if volumes <= 0:
                try:
                    total_volume = self.pool[symbol]["volume"]
                    volumes = -int(total_volume * (1 - volumes))
                except KeyError:
                    volumes = int(self.cash * volumes / unit / self.short_margin)
            else:
                volumes = int(self.cash / unit * volumes / self.long_margin)
            self.order(symbol, volumes, price)
        except Exception as e:
            print(e)

    def update_price(self, price_dic: dict[str:float]) -> None:
        for stock in self.pool.keys():
            self.pool[stock]["price"] = price_dic[stock]
            self.calculate_portfolio_value()

    def calculate_portfolio_value(self):
        """
        计算投资组合价值。计算之前需要先更新证券价格
        """
        try:
            if len(self.pool) == 0:
                return self.cash
            else:
                security_value = 0
                for symbol in self.pool.keys():
                    price = self.pool[symbol]["price"]
                    volume = self.pool[symbol]["volume"]
                    if symbol in ["IF.CFX", "IH.CFX"]:
                        unit = price * 300
                    elif symbol in ["IC.CFX", "IM.CFX"]:
                        unit = price * 200
                    margin = (
                        abs(self.pool.get(symbol, {}).get("volume", 0))
                        * unit
                        * self.long_margin
                    )
                    self.cash -= margin - self.margin
                    self.margin = margin
                    security_value += unit * volume + self.fate_cash
                self.portfolio_value = self.cash + security_value + self.margin
                return self.portfolio_value
        except Exception as e:
            print(e)


if __name__ == "__main__":
    account = futureAccount()
    print(account.cash, account.portfolio_value)
    account.order_to("111", 0.5, 6)
    print(account.pool, account.transactions)
    print(account.cash, account.portfolio_value)
    account.update_price({"111": 7})
    print(account.pool, account.transactions)
    print(account.cash, account.portfolio_value)
