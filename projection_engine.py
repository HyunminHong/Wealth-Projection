import numpy as np

class ProjectionEngine:
    def __init__(
        self,
        gross_salary1: float,
        gross_salary2: float,
        grocery: float,
        subscription: float,
        lesson: float,
        investment: float,
        cash: float,
        insurance1: float,
        insurance2: float,
        annual_rtn: float,
        zorgtoeslag: bool = False,
        huurtoeslag: bool = False,
    ):
        self.gross_salary1 = gross_salary1
        self.gross_salary2 = gross_salary2
        self.grocery = grocery
        self.subscription = subscription
        self.lesson = lesson
        self.investment = investment
        self.cash = cash
        self.insurance1 = insurance1
        self.insurance2 = insurance2
        self.annual_rtn = annual_rtn
        self.zorgtoeslag = zorgtoeslag
        self.huurtoeslag = huurtoeslag

        # Derived parameters
        self.salary1 = self.net_salary(gross_salary1)
        self.salary2 = self.net_salary(gross_salary2) + (131 if zorgtoeslag else 0)

        self.monthly_rent = (326 + 138) if huurtoeslag else (696 + 138)
        self.living_cost = (
            self.monthly_rent + grocery + subscription + lesson + insurance1 + insurance2
        )

        assert self.salary1 + self.salary2 > investment + cash + self.living_cost, "Expenses exceed income!"

        self.remainder_acc = self.salary1 + self.salary2 - (investment + cash + self.living_cost)

    def net_salary(self, gross_monthly: float) -> float:
        y = gross_monthly * 12
        b1, b2 = 38_441, 76_817
        r1, r2, r3 = 0.3582, 0.3748, 0.4950

        if y <= b1:
            tax = y * r1
        elif y <= b2:
            tax = b1 * r1 + (y - b1) * r2
        else:
            tax = b1 * r1 + (b2 - b1) * r2 + (y - b2) * r3

        if y <= 28_406:
            gen_credit = 3068.0
        elif y <= 76_817:
            gen_credit = max(0.0, 3068.0 - 0.06337 * (y - 28_406))
        else:
            gen_credit = 0.0

        if y <= 12_169:
            lab_credit = 0.08053 * y
        elif y <= 26_288:
            lab_credit = 980 + 0.3003 * (y - 12_169)
        elif y <= 43_071:
            lab_credit = 5220 + 0.02258 * (y - 26_288)
        elif y <= 129_077:
            lab_credit = max(0.0, 5599 - 0.06510 * (y - 43_071))
        else:
            lab_credit = 0.0

        net_year = y - tax + gen_credit + lab_credit
        return round(net_year / 12, 2)

    def projection_wealth(self, years: int, current_invested: float = 0.0, current_cash: float = 0.0):
        n_months = years * 12
        i = (1 + self.annual_rtn) ** (1 / 12) - 1

        inv = np.zeros(n_months)
        cash = np.zeros(n_months)

        inv[0] = (current_invested + self.investment) * (1 + i)
        cash[0] = current_cash + self.cash

        for t in range(1, n_months):
            inv[t] = (inv[t - 1] + self.investment) * (1 + i)
            cash[t] = cash[t - 1] + self.cash

        return inv, cash
    
    def mortgage_cap(self) -> float:
        return 5.0 * (self.gross_salary1 + self.gross_salary2) * 12.0

    def is_feasible_purchase(
        self,
        years_until_purchase: int,
        down_payment: tuple[float, float],
        market_value: float,
    ) -> tuple[bool, float]:
        inv_before, cash_before = self.projection_wealth(years_until_purchase)
        dp = down_payment[0] * inv_before[-1] + down_payment[1] * cash_before[-1]
        loan_amount = market_value - dp
        return (loan_amount <= self.mortgage_cap()), loan_amount

    def annuity_payment(self, loan_amount: float, mortgage_rate: float, term_years: int) -> float:
        n = term_years * 12
        i = mortgage_rate / 12
        if i == 0:
            return loan_amount / n
        return (loan_amount * i) / (1 - (1 + i) ** -n)

    def project_with_purchase(
        self,
        years_total: int,
        years_until_purchase: int,
        down_payment: tuple[float, float],
        market_value: float,
        mortgage_rate: float = 0.0379,
        mortgage_term: int = 30,
    ):
        inv_before, cash_before = self.projection_wealth(years_until_purchase)

        loan_amount = market_value - (
            down_payment[0] * inv_before[-1] + down_payment[1] * cash_before[-1]
        )
        annuity = self.annuity_payment(loan_amount, mortgage_rate, mortgage_term)

        remaining_inv = (1 - down_payment[0]) * inv_before[-1]
        remaining_cash = (1 - down_payment[1]) * cash_before[-1]

        inv_after, cash_after = self.projection_wealth(
            years_total - years_until_purchase,
            current_invested=remaining_inv,
            current_cash=remaining_cash,
        )

        inv = np.concatenate([inv_before, inv_after])
        cash = np.concatenate([cash_before, cash_after])

        return inv, cash, annuity
