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


class BREED:
    breed: str
    unit: int

    def __init__(self, breed: str, unit: int) -> None:
        self.breed = breed
        self.unit = unit


class currencyBREED(BREED):
    breed: str
    unit: int

    def __init__(self, breed: str) -> None:
        super().__init__(breed, 0)


BASE_CASH = currencyBREED("base_currency")


class asset(TypedDict):
    breed: str
    volume: int
    price: float
    unit: int


class pool:
    breed: BREED
    volume: int
    price: float
    unit: int

    def __init__(self, breed: BREED, price: float, volume: int | float = 0) -> None:
        self.breed = breed
        self.price = price
        self.unit = breed.unit
        if self.unit != 0:
            assert isinstance(volume, int) and volume > 0

        self.volume = volume if self.unit > 0 else float(volume)
        self.cash_value = self.price * self.volume
        self.cost = price * self.volume

    @property
    def dic(self) -> asset:
        pool_dic: asset = {
            "breed": self.breed.breed,
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
        pool_values = [self.breed.breed, self.volume, self.price, self.unit]
        return pool_values

    @property
    def aver_cost(self) -> float:
        try:
            aver_cost = round(self.cost / self.volume, accuracy)
            return aver_cost
        except ZeroDivisionError:
            return 0.0

    @property
    def returns(self) -> float:
        return self.cash_value - self.cost

    def __call__(self) -> asset:
        pool_dic: asset = {
            "breed": self.breed.breed,
            "volume": self.volume,
            "price": self.price,
            "unit": self.unit,
        }
        return pool_dic

    def __repr__(self) -> str:
        return f"pool(breed={self.breed.breed},volume={self.volume},price={self.price},unit={self.unit})"

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
        self.cost -= amount * self.price
        self.cal_value()

    def save(self, amount: float | int):
        """
        add asset
        """
        assert amount >= 0, "using save method need amount no less than 0."
        amount = float(amount) if self.unit == 0 else int(amount)
        self.volume += amount
        self.cost += amount * self.price
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
        save_amount = amount - volume
        self.save(save_amount)
        return save_amount

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
        return self.save_to(amount)

    def withdraw_to(
        self, amount: float | int, integral: Literal["ceil", "floor"] = "ceil"
    ):
        assert (
            self.volume >= amount
        ), f"current volume {self.volume} already less than target amount {amount}"
        amount = float(amount) if self.unit == 0 else self.inte(amount, integral)
        volume = self.volume
        withdraw_amount = volume - amount
        self.withdraw(withdraw_amount)
        return withdraw_amount

    def withdraw_to_rate(
        self, rate: float, integral: Literal["ceil", "floor"] = "ceil"
    ):
        """
        withdraw asset to a certain rate
        """
        amount = round(self.volume * rate, accuracy)
        amount = float(amount) if self.unit == 0 else self.inte(amount, integral)
        return self.withdraw_to(amount)

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
        cur_breed = currencyBREED(breed)
        super().__init__(cur_breed, price, volume)


class cashPool(pool):
    def __init__(self, volume: float = 0.0) -> None:
        price = 1.0
        breed = BASE_CASH
        super().__init__(breed, price, volume)


class shortSellCP(currencyPool):

    def __init__(self, volume: float = 0.0) -> None:
        price = 1.0
        super().__init__("base_currency", price, volume)

    def withdraw(self, amount: float | int):
        """
        withdraw certain amount asset, can short sell
        """
        assert amount >= 0, "using withdraw method need amount no less than 0."
        self.volume -= amount
        self.cal_value()

    def check(self, amount: float | int):
        return amount


class baseAccount(ABC):
    pools: dict[str, pool]
    base: float
    portfolio_value: float

    def __init__(
        self,
        base_currency: float,
        pools: dict[str, pool] = {},
    ) -> None:
        self.pools = pools
        cash = cashPool(base_currency)
        self.pools["base_currency"] = cash
        self.cal_portfolio_value()
        self.base = self.portfolio_value

    @property
    def pfo_return(self):
        """
        the portfolio return
        """
        return self.portfolio_value - self.base

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
        amount = self.pools[breed].check(amount)
        return amount

    def _withdraw_breed(self, breed: BREED, amount: float | int):
        """
        withdraw breed from pools
        """
        try:
            poo = self.pools[breed.breed]
            poo.withdraw(amount)
            if self.pools[breed.breed].volume == 0:
                del self.pools[breed.breed]
            self.cal_portfolio_value()
        except KeyError:
            pass

    def _save_breed(
        self,
        breed: BREED,
        amount: float | int,
        price: float,
    ):
        """
        save breed in pools
        """
        try:
            position = pool(breed, price, amount)
            poo = self.pools[breed.breed]
            poo += position
        except KeyError:
            self.pools[breed.breed] = position
        except Exception as e:
            print(e)
        finally:
            self.cal_portfolio_value()

    def handle_cash(self, amount: float):
        """
        use to add or reduce account cash and base.
        if amount >=0, withdraw cash, else save.
        """
        breed = currencyBREED("base_currency")
        if amount >= 0:
            self._withdraw_breed(breed, amount)
        else:
            self._save_breed(breed, -amount, 1.0)
        self.base += amount

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

    def _trade(
        self, breed_A: BREED, breed_B: BREED, orin_volume_A: int, price_B: float
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
            code = -1
            price_A = self.get_breed_price(breed_A.breed)
            unit_A = self.pools[breed_A.breed].breed.unit
            volume_A = self.check_breed(breed_A.breed, orin_volume_A)
            unit = breed_B.unit
            volume_B = self.can_buy_volume(price_A, unit_A, volume_A, price_B, unit)
            if volume_B < unit:
                return code
            volume_A = self.can_buy_volume(price_B, unit, volume_B, price_A, unit_A)
            self._withdraw_breed(breed_A, volume_A)
            self._save_breed(breed_B, volume_B, price_B)
            if volume_A == orin_volume_A:
                code = 1
            elif 0 < volume_A and volume_A < orin_volume_A:
                code = 0
            else:
                code = -1
        except KeyError:
            pass
        finally:
            return code

    def get_breed_price(self, breed: str) -> float:
        try:
            price = self.pools[breed].price
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
                data[col] = [self.pools[bre].dic[col] for bre in breeds]
            data["return"] = [self.pools[bre].returns for bre in breeds]
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


class unFeeAaccount(baseAccount):
    """
    can not short selling and without transaction fee
    Due to the characteristics of Python floating-point operations, the control accuracy is 6 decimal places
    """

    def buy(self, breed: BREED, volume: int | float, price: float):
        assert volume >= 0
        volume = volume if breed.unit == 0 else int(volume)
        need_money = volume * price
        code = self._trade(BASE_CASH, breed, need_money, price)
        return code

    def buy_to(self, breed: BREED, rate: float, price: float):
        assert rate > 0
        cash = self.pools["base_currency"].volume
        need_money = cash * rate
        code = self._trade(BASE_CASH, breed, need_money, price)
        return code

    def sell(self, breed: BREED, volume: int | float, price: float):
        assert volume >= 0
        self.update_price(breed.breed, price)
        code = self._trade(breed, BASE_CASH, volume, 1.0)
        return code

    def sell_to(self, breed: BREED, rate: float, price: float):
        assert rate >= 0
        try:
            code = -1
            assert breed.breed in self.pools.keys()
            self.update_price(breed.breed, price)
            rate = round(1 - rate, accuracy)
            volume = self.pools[breed.breed].volume * rate
            volume = volume if breed.unit == 0 else int(volume)
            code = self.sell(breed, volume, price)
        except AssertionError:
            print(f"Didn't have {breed} in pools, can't sell")
        finally:
            return code

    def order(self, breed: BREED, volume: int, price: float):
        volume = volume if breed.unit == 0 else int(volume)
        if volume >= 0:
            code = self.buy(breed, volume, price)
        else:
            code = self.sell(breed, -volume, price)
        return code

    def order_to(self, breed: BREED, rate: int, price: float):
        try:
            code = -1
            assert -1 <= rate and rate <= 1
            if rate > 0:
                code = self.buy_to(breed, rate, price)
            else:
                code = self.sell_to(breed, -rate, price)
        except AssertionError:
            print("Warning: order rate must belong to [0,1]")
        finally:
            return code

    def cal_portfolio_value(self):
        porfolio_value = 0
        for poo in self.pools.values():
            porfolio_value += poo.cash_value
        self.portfolio_value = porfolio_value
        return porfolio_value


breed_CF = BREED("CF", 1)
breed_CH = BREED("CH", 1)
poo = pool(breed_CF, 5000, 10)
print(poo.aver_cost, poo.returns)
acc = unFeeAaccount(1000000, {"CF": poo})
print(acc.portfolio_value)
acc.buy(breed_CF, 1, 5200)
acc.sell_to(breed_CF, 0.5, 5000)
acc.buy(breed_CH, 20, 2000)
acc.update_price(breed_CH.breed, 1800)
print(acc.portfolio_value)
print(acc.get_pools_df())
print(acc.pfo_return)
