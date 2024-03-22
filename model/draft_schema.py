from math import gcd, ceil, floor
from abc import ABC, abstractmethod
from typing import Literal, TypedDict
from collections import defaultdict

import pandas as pd

accuracy = 6


def lcm_float(x, y) -> float:
    multiplier = 10 ** max(len(str(x).split(".")[-1]), len(str(y).split(".")[-1]))
    x_int = int(x * multiplier)
    y_int = int(y * multiplier)
    lcm_int = x_int * y_int // gcd(x_int, y_int)
    return round(lcm_int / multiplier, accuracy)


def float_division(x, y) -> float:
    multiplier = 10 ** max(len(str(x).split(".")[1]), len(str(y).split(".")[1]))
    x_int = int(x * multiplier)
    y_int = int(y * multiplier)
    result_int = x_int // y_int
    result_float = result_int / multiplier
    return result_float


class asset(TypedDict):
    breed: str
    volume: int
    price: float
    unit: int


class pool:
    breed: str
    volume: int
    price: float
    unit: int

    def __init__(
        self, breed: str, price: float, unit: int, volume: int | float = 0
    ) -> None:
        self.breed = breed
        self.price = price
        self.unit = unit
        if self.unit != 0:
            assert isinstance(volume, int) and volume > 0

        self.volume = volume if self.unit > 0 else float(volume)
        self.cash_value = self.price * self.volume

    @property
    def dic(self) -> asset:
        pool_dic: asset = {
            "breed": self.breed,
            "volume": self.volume,
            "price": self.price,
            "unit": self.unit,
        }
        return pool_dic

    @property
    def keys(self) -> list:
        pool_keys = ["breed", "colume", "price", "unit"]
        return pool_keys

    @property
    def values(self) -> list:
        pool_values = [self.breed, self.volume, self.price, self.unit]
        return pool_values

    def __call__(self) -> asset:
        pool_dic: asset = {
            "breed": self.breed,
            "volume": self.volume,
            "price": self.price,
            "unit": self.unit,
        }
        return pool_dic

    def __repr__(self) -> str:
        return f"pool(breed={self.breed},volume={self.volume},price={self.price},unit={self.unit})"

    def __str__(self) -> str:
        return str(self.dic)

    def __add__(self, poo):
        assert isinstance(poo, pool)
        assert self.breed == poo.breed, "Warning: just the same breed can be added!"
        self.update_price(poo.price)
        self.save(poo.volume)
        return self

    def __sub__(self, poo):
        assert isinstance(poo, pool)
        assert self.breed == poo.breed, "Warning: just the same breed can be added!"
        self.update_price(poo.price)
        self.withdraw(poo.volume)
        return self

    def cal_value(self) -> float:
        value = round(self.price * self.volume, accuracy)
        self.cash_value = value
        return value

    def update_price(self, price: float):
        """
        update price if price change
        """
        assert price > 0, "price need more than 0"
        self.price = price
        self.cal_value()

    def inte(self, amount: float, integral: Literal["ceil", "floor"] = "ceil"):
        integ = ceil if integral == "ceil" else floor
        return integ(amount)

    def check(self, amount: float | int):
        """
        check the enough amout can be withdrawed
        """
        amount = float(amount) if self.unit == 0 else int(amount)
        if self.volume >= amount:
            return amount
        else:
            print(f"current volume is {self.volume}, got {amount}")
            return self.volume

    def withdraw(self, amount: float | int):
        """
        withdraw certain amount asset
        """
        assert amount >= 0, "using withdraw method need amount no less than 0."
        amount = self.check(amount)
        self.volume -= amount
        self.cal_value()

    def save(self, amount: float | int):
        """
        add asset
        """
        assert amount >= 0, "using save method need amount no less than 0."
        amount = float(amount) if self.unit == 0 else int(amount)
        self.volume += amount
        self.cal_value()

    def save_to(self, amount: float | int, integral: Literal["ceil", "floor"] = "ceil"):
        """
        save in asset to a certain volume
        """
        assert (
            self.volume <= amount
        ), f"current volume {self.volume} already more than target amount {amount}"
        amount = float(amount) if self.unit == 0 else self.inte(amount, integral)
        volume = self.volume
        self.volume = amount
        self.cal_value()
        return amount - volume

    def save_as_rate(self, rate: float, integral: Literal["ceil", "floor"] = "ceil"):
        """
        save in asset as a certain rate
        """
        amount = round(self.volume * rate, accuracy)
        amount = float(amount) if self.unit == 0 else self.inte(amount, integral)
        self.save(amount)
        return amount

    def save_to_rate(self, rate: float, integral: Literal["ceil", "floor"] = "ceil"):
        """
        save asset to a certain rate
        """
        assert (
            rate >= 1.0
        ), f"save to a certain rate nee rate no less then 1, got {rate}"
        amount = round(self.volume * rate, accuracy)
        amount = float(amount) if self.unit == 0 else self.inte(amount, integral)
        volume = self.volume
        self.volume = amount
        self.cal_value()
        return amount - volume

    def withdraw_to(
        self, amount: float | int, integral: Literal["ceil", "floor"] = "ceil"
    ):
        assert (
            self.volume >= amount
        ), f"current volume {self.volume} already less than target amount {amount}"
        amount = float(amount) if self.unit == 0 else self.inte(amount, integral)
        volume = self.volume
        self.volume = amount
        self.cal_value()
        return volume - amount

    def withdraw_to_rate(
        self, rate: float, integral: Literal["ceil", "floor"] = "ceil"
    ):
        """
        withdraw asset to a certain rate
        """
        amount = round(self.volume * rate, accuracy)
        amount = float(amount) if self.unit == 0 else self.inte(amount, integral)
        volume = self.volume
        self.volume = amount
        self.cal_value()
        return volume - amount

    def withdraw_as_rate(
        self, rate: float, integral: Literal["ceil", "floor"] = "floor"
    ):
        """
        withdraw asset as a certain rate
        """
        amount = round(self.volume * rate, accuracy)
        amount = float(amount) if self.unit == 0 else self.inte(amount, integral)
        self.withdraw(amount)
        return amount


class currencyPool(pool):

    def __init__(self, breed: str, price: float, volume: float = 0.0) -> None:
        unit = 0
        super().__init__(breed, price, unit, volume)


class cashPool(currencyPool):
    def __init__(self, volume: float = 0.0) -> None:
        breed = "base_currency"
        price = 1.0
        super().__init__(breed, price, volume)


class account(ABC):
    pools: dict[str, pool]
    portfolio_value: float

    def __init__(
        self, base_currency: float, pools: dict[str, pool] = defaultdict(pool)
    ) -> None:
        self.pools = pools
        cash = cashPool(base_currency)
        self.pools["base_currency"] = cash
        self.cal_portfolio_value()

    def count_volume(self, breed: str) -> int | float:
        """
        count the position of certain breed
        """
        try:
            poo = self.pools[breed]
            return poo.volume
        except KeyError:
            print(f"Warning: the breed {breed} not in pools")
            return 0

    def count_base_current(self) -> float:
        """
        count the cash
        """
        return self.count_volume("base_currency")

    def check_cash(self, amount: float | None = None) -> float:
        """
        check the cash wether is enough
        """
        try:
            if amount is None:
                return self.pools["base_currency"].volume
            cash = self.pools["base_currency"].check(amount)
            return cash
        except KeyError:
            print("Doese not have base current")
            return 0.0

    def check_breed(self, breed: str, amount: float | int | None = None) -> float:
        volume = self.count_volume(breed)
        if amount is None:
            return volume
        volume = self.pools[breed].check(amount)
        return volume

    def withdraw_breed(self, breed: str, amount: float | int):
        """
        withdraw breed from pools
        """
        try:
            poo = self.pools[breed]
            poo.withdraw(amount)
            if self.pools[breed].volume == 0:
                del self.pools[breed]
            self.cal_portfolio_value()
        except KeyError:
            pass

    def save_breed(
        self,
        breed: str,
        amount: float | int,
        price: float,
        unit: int = 1,
    ):
        """
        save breed in pools
        """
        try:
            self.pools[breed].save(amount)
            self.update_price(breed, price)
        except KeyError:
            position = pool(breed, price, unit, amount)
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
            self.save_breed("base_currency", -amount, 1.0, 0)

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
            self.pools[breed].update_price(price)
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


poo: pool = {"breed": "CF", "price": 5000, "unit": 1, "volume": 10}
acc = unFeeAaccount(1000000, {"CF": poo})
print(acc.portfolio_value)
acc.trade("CF", "CH", 3, 2000, unit=0)
acc.trade("CH", "base_currency", 1, 1, 1)
print(acc.portfolio_value)
print(acc.get_pools_df())
