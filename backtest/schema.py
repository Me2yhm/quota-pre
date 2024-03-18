from datetime import datetime, timedelta
from collections import defaultdict


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
