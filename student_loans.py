#!/usr/bin/env python3
"""
student_loans.py -- effective-rate math and payoff simulation for federal student loans.

Why this exists
---------------
Federal student loans accrue SIMPLE daily interest on principal only:

    daily accrual = principal * annual_rate / 365

Outstanding accrued-but-unpaid interest does NOT itself accrue. So if you owe
$139,000 of which $9,000 is accrued interest, only $130,000 is actually
generating interest. Two consequences:

  1. The effective rate on your TOTAL balance is lower than the
     principal-weighted average of your loan rates, by exactly the factor
     principal / (principal + accrued_interest).

  2. More usefully: the accrued-interest bucket is not a low-rate chunk, it is
     a ZERO-rate chunk. Dollars you pay into it reduce your balance but reduce
     your future accrual by nothing. Only dollars that reach principal earn you
     the loan's rate. `summary` prints this as a marginal-rate schedule.

Caveat that breaks (2): CAPITALIZATION. If accrued interest capitalizes it
becomes principal and starts accruing. The 2023 regulations eliminated most
capitalization triggers, but consolidation and exiting certain forbearances
still capitalize. Use `Loan.capitalize()` / `--capitalize-on` to model it.

Conventions
-----------
* Day count: actual days / 365 (no leap-day adjustment), which is what federal
  servicers use.
* Rates are entered as percentages (5.5) or decimals (0.055); anything > 1.0 is
  read as a percentage. Enter the rate you are actually charged, i.e. AFTER any
  autopay discount, and use --rate-change to model the discount expiring.
* Payments apply per loan in the order: accrued interest, then principal. This
  matches standard federal servicer allocation (fees are not modeled).
* `proportional` is the servicer default: the payment is split across loans by
  share of balance, which drains the accrued-interest backlog first.
* Every strategy pays the servicer's required minimum on each loan first --
  that is a constraint, not a choice. Strategies differ only in where the
  EXCESS above the minimums goes: `avalanche` (highest rate), `snowball`
  (smallest balance), `custom` (a priority list you supply), or `proportional`
  (spread by balance share, the servicer default). Real minimums usually run
  well above monthly interest, which forces the accrued-interest backlog down
  faster than you'd choose; without a minimum_payment column the floor falls
  back to interest-only, which overstates how long the 0% backlog can be held.

Stdlib only. Python 3.10+.

Usage
-----
    python student_loans.py template > loans.csv     # write a starter CSV
    python student_loans.py summary  loans.csv
    python student_loans.py simulate loans.csv --payment 2500
    python student_loans.py simulate loans.csv --payment 2500 --priority AH,AG
    python student_loans.py compare  loans.csv --payment 2500 \
        --schedule-out avalanche.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Iterable, Sequence

DAYS_PER_YEAR = 365.0
CENT = 0.005  # treat balances below half a cent as paid off


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass
class Loan:
    """A single federal loan (usually one disbursement)."""

    name: str
    principal: float
    rate: float  # annual, as a decimal (0.055 == 5.5%)
    accrued_interest: float = 0.0
    minimum_payment: float = 0.0  # servicer's required monthly payment
    rate_reduction_expires: date | None = None  # when a temporary discount ends
    as_of: date | None = None  # date these balances were pulled

    def __post_init__(self) -> None:
        if self.principal < 0:
            raise ValueError(f"{self.name}: principal cannot be negative")
        if self.accrued_interest < 0:
            raise ValueError(f"{self.name}: accrued interest cannot be negative")
        if not 0 <= self.rate < 1:
            raise ValueError(
                f"{self.name}: rate {self.rate} is out of range; "
                "pass 5.5 for 5.5% or 0.055 as a decimal"
            )

    @property
    def balance(self) -> float:
        """Total payoff amount today: principal plus outstanding interest."""
        return self.principal + self.accrued_interest

    @property
    def daily_accrual(self) -> float:
        return self.principal * self.rate / DAYS_PER_YEAR

    @property
    def annual_accrual(self) -> float:
        return self.principal * self.rate

    def accrue(self, days: int) -> float:
        """Accrue `days` of simple interest on principal. Returns amount accrued."""
        amount = self.principal * self.rate * days / DAYS_PER_YEAR
        self.accrued_interest += amount
        return amount

    def capitalize(self) -> float:
        """Roll outstanding interest into principal. Returns amount capitalized."""
        amount = self.accrued_interest
        self.principal += amount
        self.accrued_interest = 0.0
        return amount

    def pay(self, amount: float) -> tuple[float, float]:
        """Apply `amount`, interest first then principal.

        Returns (interest_paid, principal_paid). Never overpays the balance.
        """
        if amount <= 0:
            return 0.0, 0.0
        to_interest = min(amount, self.accrued_interest)
        self.accrued_interest -= to_interest
        to_principal = min(amount - to_interest, self.principal)
        self.principal -= to_principal
        return to_interest, to_principal

    @property
    def paid_off(self) -> bool:
        return self.balance <= CENT


@dataclass(frozen=True)
class RateChange:
    """A scheduled change in rate, in percentage points, applied to all loans.

    The motivating case is a 0.25%/1.00% autopay interest-rate discount
    expiring: `RateChange(date(2028, 6, 1), +1.0)`.
    """

    effective: date
    delta_pct_points: float
    only: tuple[str, ...] | None = None  # loan names, or None for all

    def apply(self, loans: Iterable[Loan]) -> None:
        for loan in loans:
            if self.only is None or loan.name in self.only:
                loan.rate = max(0.0, loan.rate + self.delta_pct_points / 100.0)


@dataclass(frozen=True)
class PaymentStep:
    """A planned change in the monthly payment, effective on a date."""

    effective: date
    amount: float


def payment_on(base: float, steps: Sequence[PaymentStep], when: date) -> float:
    """The monthly payment in force on `when`: the latest step at or before it."""
    amount = base
    for step in sorted(steps, key=lambda s: s.effective):
        if step.effective <= when:
            amount = step.amount
    return amount


# --------------------------------------------------------------------------
# Parsing / IO
# --------------------------------------------------------------------------

TEMPLATE = """\
name,principal,rate,accrued_interest
# One row per loan (usually one per disbursement). Lines starting with # are
# ignored. `rate` is the rate you're actually charged today -- include any
# autopay discount, then model its expiry with --rate-change.
# `principal` is the accruing balance; `accrued_interest` is outstanding
# unpaid interest, which does NOT accrue. Both appear on your servicer's
# loan detail page. If you only have a combined balance, put it all in
# `principal` and leave accrued_interest at 0 -- the numbers will be slightly
# pessimistic but never wrong in the other direction.
Unsub 2019-1,10500,4.53,412.18
Unsub 2019-2,10500,4.53,405.90
Grad PLUS 2020-1,18000,6.28,721.44
Grad PLUS 2020-2,18000,6.28,715.02
Unsub 2021-1,10500,3.73,240.11
"""


def parse_rate(value: str | float) -> float:
    """Accept 5.5, '5.5', '5.5%', or 0.055 and return a decimal rate."""
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    number = float(value)
    return number / 100.0 if number > 1.0 else number


# Accept whatever your servicer's export calls things.
ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name", "loan_id", "id", "loan"),
    "principal": ("principal", "principal_balance", "balance", "current_principal"),
    "rate": ("rate", "interest_rate", "apr", "effective_rate"),
    "accrued_interest": ("accrued_interest", "interest", "accrued", "unpaid_interest"),
    "minimum_payment": ("minimum_payment", "minimum", "min_payment", "monthly_payment"),
    "rate_reduction_expires": ("rate_reduction_expires", "reduction_expires", "discount_expires"),
    "as_of": ("as_of", "as_of_date", "date", "statement_date"),
}


def _pick(row: dict[str, str], field_name: str) -> str | None:
    for alias in ALIASES[field_name]:
        if row.get(alias) not in (None, ""):
            return row[alias]
    return None


def _num(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(str(value).replace(",", "").replace("$", "").strip())


def _day(value: str | None) -> date | None:
    if value in (None, ""):
        return None
    return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()


def load_loans(path: str) -> list[Loan]:
    """Read loans from CSV. Blank lines and #-comments are skipped.

    Column names are matched case-insensitively against ALIASES, so a servicer
    export using loan_id / principal_balance / interest_rate loads unchanged.
    """
    loans: list[Loan] = []
    with open(path, newline="", encoding="utf-8") as handle:
        lines = (line for line in handle if line.strip() and not line.lstrip().startswith("#"))
        for lineno, raw in enumerate(csv.DictReader(lines), start=2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            missing = [f for f in ("name", "principal", "rate") if _pick(row, f) is None]
            if missing:
                raise ValueError(
                    f"{path}: no column found for {missing}; "
                    f"accepted names: { {f: ALIASES[f] for f in missing} }"
                )
            try:
                loans.append(
                    Loan(
                        name=str(_pick(row, "name")),
                        principal=_num(_pick(row, "principal")),
                        rate=parse_rate(str(_pick(row, "rate"))),
                        accrued_interest=_num(_pick(row, "accrued_interest")),
                        minimum_payment=_num(_pick(row, "minimum_payment")),
                        rate_reduction_expires=_day(_pick(row, "rate_reduction_expires")),
                        as_of=_day(_pick(row, "as_of")),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"{path} row {lineno}: {exc}") from exc
    if not loans:
        raise ValueError(f"{path}: no loan rows found")

    stamps = {l.as_of for l in loans if l.as_of}
    if len(stamps) > 1:
        print(
            f"warning: rows carry different as-of dates {sorted(stamps)}; "
            "accrued interest is a snapshot, so mixing dates skews the starting balance.",
            file=sys.stderr,
        )
    return loans


def auto_rate_changes(loans: Sequence[Loan], delta_pct_points: float = 1.0) -> list[RateChange]:
    """Turn per-loan `rate_reduction_expires` dates into RateChange events."""
    by_date: dict[date, list[str]] = {}
    for loan in loans:
        if loan.rate_reduction_expires:
            by_date.setdefault(loan.rate_reduction_expires, []).append(loan.name)
    everyone = {l.name for l in loans}
    return [
        RateChange(when, delta_pct_points, None if set(names) == everyone else tuple(names))
        for when, names in sorted(by_date.items())
    ]


def parse_payment_step(spec: str) -> PaymentStep:
    """Parse 'YYYY-MM-DD:2500'."""
    try:
        when, amount = spec.split(":")
        return PaymentStep(datetime.strptime(when, "%Y-%m-%d").date(), float(amount))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"bad --payment-step {spec!r}; expected DATE:AMOUNT"
        ) from exc


def parse_rate_change(spec: str) -> RateChange:
    """Parse 'YYYY-MM-DD:+1.0' or 'YYYY-MM-DD:+1.0:LoanA|LoanB'."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"bad --rate-change {spec!r}; expected DATE:DELTA[:Loan|Loan]"
        )
    try:
        when = datetime.strptime(parts[0], "%Y-%m-%d").date()
        delta = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad --rate-change {spec!r}: {exc}") from exc
    only = tuple(p.strip() for p in parts[2].split("|")) if len(parts) == 3 else None
    return RateChange(when, delta, only)


# --------------------------------------------------------------------------
# Rate math (no simulation)
# --------------------------------------------------------------------------


@dataclass
class RateSummary:
    principal: float
    accrued_interest: float
    total_balance: float
    weighted_average_rate: float  # principal-weighted; the rate you're charged
    simple_average_rate: float  # unweighted mean, shown only to contrast
    effective_rate_on_balance: float  # annual accrual / total balance
    daily_accrual: float
    annual_accrual: float
    tranches: list[tuple[str, float, float]]  # (label, dollars, marginal rate)
    total_minimum: float = 0.0
    as_of: date | None = None


def rate_summary(loans: Sequence[Loan]) -> RateSummary:
    principal = sum(l.principal for l in loans)
    accrued = sum(l.accrued_interest for l in loans)
    total = principal + accrued
    annual = sum(l.annual_accrual for l in loans)

    weighted = annual / principal if principal else 0.0
    simple = sum(l.rate for l in loans) / len(loans)
    effective = annual / total if total else 0.0

    # Marginal-rate schedule: the order in which dollars retire debt under an
    # avalanche strategy, and what each dollar actually saves you.
    tranches: list[tuple[str, float, float]] = []
    if accrued > CENT:
        tranches.append(("accrued interest (all loans)", accrued, 0.0))
    for loan in sorted(loans, key=lambda l: -l.rate):
        if loan.principal > CENT:
            tranches.append((f"principal: {loan.name}", loan.principal, loan.rate))

    return RateSummary(
        principal=principal,
        accrued_interest=accrued,
        total_balance=total,
        weighted_average_rate=weighted,
        simple_average_rate=simple,
        effective_rate_on_balance=effective,
        daily_accrual=annual / DAYS_PER_YEAR,
        annual_accrual=annual,
        tranches=tranches,
        total_minimum=sum(l.minimum_payment for l in loans),
        as_of=max((l.as_of for l in loans if l.as_of), default=None),
    )


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

STRATEGIES = ("avalanche", "proportional", "snowball", "custom")


def add_months(d: date, n: int) -> date:
    """Add n months, clamping the day (Jan 31 + 1 month -> Feb 28/29)."""
    month_index = d.month - 1 + n
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    days_in_month = (date(year + (month == 12), month % 12 + 1, 1) - date(year, month, 1)).days
    return date(year, month, min(d.day, days_in_month))


def required_floor(loans: Sequence[Loan], period_accrual: Sequence[float]) -> list[float]:
    """The non-negotiable payment on each loan this period.

    That's the servicer's required minimum, or -- if no minimum is on file --
    this period's interest, which is the least you can pay without the balance
    growing. Capped at the payoff amount so a nearly-dead loan doesn't demand
    its full minimum.
    """
    return [
        min(max(loan.minimum_payment, accrual), loan.balance)
        for loan, accrual in zip(loans, period_accrual)
    ]


def _pay_floor(
    loans: Sequence[Loan], budget: float, floor: Sequence[float]
) -> tuple[float, float, float]:
    """Pay each loan its required floor. Returns (interest, principal, leftover).

    Paid pro rata if the budget can't cover the total, which is the point at
    which you'd actually be delinquent.
    """
    owed = sum(floor)
    if owed <= CENT:
        return 0.0, 0.0, budget
    ratio = min(1.0, budget / owed)
    interest_paid = principal_paid = 0.0
    for amount, loan in zip(floor, loans):
        i, p = loan.pay(amount * ratio)
        interest_paid += i
        principal_paid += p
    return interest_paid, principal_paid, max(0.0, budget - interest_paid - principal_paid)


def _spread(loans: Sequence[Loan], budget: float) -> tuple[float, float]:
    """Split `budget` across loans by share of balance, redistributing as loans clear."""
    interest_paid = principal_paid = 0.0
    remaining = budget
    for _ in range(len(loans) + 1):
        active = [l for l in loans if not l.paid_off]
        total = sum(l.balance for l in active)
        if remaining <= CENT or not active or total <= CENT:
            break
        spent = 0.0
        for loan in active:
            share = min(remaining * loan.balance / total, loan.balance)
            i, p = loan.pay(share)
            interest_paid += i
            principal_paid += p
            spent += i + p
        if spent <= CENT:
            break
        remaining -= spent
    return interest_paid, principal_paid


def _allocate(
    loans: Sequence[Loan],
    payment: float,
    strategy: str,
    floor: Sequence[float],
    priority: Sequence[str] = (),
) -> tuple[float, float]:
    """Apply one payment across loans. Returns (interest_paid, principal_paid).

    Every strategy pays the required floor on every loan first -- that part is
    not a choice. The strategies differ only in where the EXCESS goes:
      proportional -- spread by balance share (the servicer default)
      avalanche    -- all of it at the highest-rate loan
      snowball     -- all of it at the smallest balance
      custom       -- all of it at the loans named in `priority`, in order,
                      then avalanche for anything not named
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")

    interest_paid, principal_paid, remaining = _pay_floor(loans, payment, floor)

    if strategy == "proportional":
        i, p = _spread(loans, remaining)
        return interest_paid + i, principal_paid + p

    if strategy == "avalanche":
        order = sorted(loans, key=lambda l: (-l.rate, l.balance))
    elif strategy == "custom":
        rank = {name: i for i, name in enumerate(priority)}
        order = sorted(loans, key=lambda l: (rank.get(l.name, len(rank)), -l.rate, l.balance))
    else:  # snowball
        order = sorted(loans, key=lambda l: (l.balance, -l.rate))

    for loan in order:
        if remaining <= CENT:
            break
        i, p = loan.pay(remaining)
        interest_paid += i
        principal_paid += p
        remaining -= i + p
    return interest_paid, principal_paid


@dataclass
class SimResult:
    strategy: str
    monthly_payment: float
    start: date
    payoff_date: date | None
    months: int
    total_paid: float
    total_interest: float  # total_paid minus starting principal; capitalization-proof
    starting_balance: float
    starting_principal: float
    total_capitalized: float
    effective_apr: float | None  # IRR of the cashflows, annualized
    negative_amortization: bool
    payment_steps: tuple = ()
    below_minimums: bool = False
    loan_payoff: dict = field(default_factory=dict)  # loan name -> date it cleared
    rows: list[dict] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.payoff_date is not None


def simulate(
    loans: Sequence[Loan],
    monthly_payment: float,
    strategy: str = "avalanche",
    start: date | None = None,
    rate_changes: Sequence[RateChange] | None = None,
    capitalize_on: Sequence[date] = (),
    max_months: int = 600,
    priority: Sequence[str] = (),
    payment_steps: Sequence[PaymentStep] = (),
) -> SimResult:
    """Run a month-by-month payoff simulation.

    Interest accrues daily on principal between payment dates; one payment is
    applied on the same day-of-month as `start`, beginning one month out.

    `start` defaults to the loans' as-of date (not today), since that is when
    the accrued-interest figures were true. `rate_changes` defaults to whatever
    the loans' `rate_reduction_expires` columns imply; pass `()` for none.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    unknown = set(priority) - {l.name for l in loans}
    if unknown:
        raise ValueError(f"--priority names loans not in the file: {sorted(unknown)}")
    if monthly_payment <= 0:
        raise ValueError("monthly payment must be positive")

    book = [replace(l) for l in loans]  # work on a copy
    if start is None:
        start = max((l.as_of for l in loans if l.as_of), default=None) or date.today()
    if rate_changes is None:
        rate_changes = auto_rate_changes(loans)
    starting_balance = sum(l.balance for l in book)
    starting_principal = sum(l.principal for l in book)

    pending_changes = sorted(rate_changes, key=lambda rc: rc.effective)
    pending_caps = sorted(capitalize_on)

    rows: list[dict] = []
    payments: list[float] = []
    loan_payoff: dict[str, date] = {}
    total_paid = total_capitalized = 0.0
    negative_am = below_minimums = False
    payoff_date: date | None = None
    cursor = start

    for month in range(1, max_months + 1):
        period_end = add_months(start, month)

        # Accrue day by day, breaking at rate-change and capitalization dates.
        events: list[tuple[date, str, object]] = []
        events += [(rc.effective, "rate", rc) for rc in pending_changes if cursor < rc.effective <= period_end]
        events += [(d, "cap", None) for d in pending_caps if cursor < d <= period_end]
        events.sort(key=lambda e: e[0])

        accrued_this_period = 0.0
        # Per-loan accrual still sitting as interest, used as the payment floor.
        floor = [0.0] * len(book)
        for when, kind, payload in events:
            days = (when - cursor).days
            for i, loan in enumerate(book):
                amount = loan.accrue(days)
                floor[i] += amount
                accrued_this_period += amount
            if kind == "rate":
                payload.apply(book)  # type: ignore[union-attr]
            else:
                total_capitalized += sum(loan.capitalize() for loan in book)
                floor = [0.0] * len(book)  # it's principal now, not a floor
            cursor = when
        days = (period_end - cursor).days
        for i, loan in enumerate(book):
            amount = loan.accrue(days)
            floor[i] += amount
            accrued_this_period += amount
        cursor = period_end

        outstanding = sum(l.balance for l in book)
        payment = min(payment_on(monthly_payment, payment_steps, period_end), outstanding)
        if payment < accrued_this_period - CENT:
            negative_am = True

        required = required_floor(book, floor)
        if payment < sum(required) - CENT:
            below_minimums = True

        interest_paid, principal_paid = _allocate(book, payment, strategy, required, priority)

        for loan in book:
            if loan.paid_off and loan.name not in loan_payoff:
                loan_payoff[loan.name] = period_end
        actually_paid = interest_paid + principal_paid
        total_paid += actually_paid
        payments.append(actually_paid)

        rows.append(
            {
                "month": month,
                "date": period_end.isoformat(),
                "payment": round(actually_paid, 2),
                "interest_accrued": round(accrued_this_period, 2),
                "interest_paid": round(interest_paid, 2),
                "principal_paid": round(principal_paid, 2),
                "principal_remaining": round(sum(l.principal for l in book), 2),
                "interest_remaining": round(sum(l.accrued_interest for l in book), 2),
                "balance_remaining": round(sum(l.balance for l in book), 2),
            }
        )

        if sum(l.balance for l in book) <= CENT:
            payoff_date = period_end
            break

    return SimResult(
        strategy=strategy,
        monthly_payment=monthly_payment,
        start=start,
        payoff_date=payoff_date,
        months=len(rows),
        total_paid=total_paid,
        # Every dollar paid beyond the original principal is interest, whether
        # or not it was capitalized into principal along the way. Summing the
        # per-payment interest allocations would undercount capitalized interest,
        # since after capitalization those dollars are booked as principal.
        # Meaningful once the loan is paid off; see `rows` for interest paid to date.
        total_interest=max(0.0, total_paid - starting_principal),
        starting_balance=starting_balance,
        starting_principal=starting_principal,
        total_capitalized=total_capitalized,
        effective_apr=irr(starting_balance, payments) if payoff_date else None,
        negative_amortization=negative_am,
        payment_steps=tuple(payment_steps),
        below_minimums=below_minimums,
        loan_payoff=loan_payoff,
        rows=rows,
    )


def irr(present_value: float, payments: Sequence[float]) -> float | None:
    """Annualized IRR of borrowing `present_value` and repaying `payments` monthly.

    This is the honest answer to "what rate am I really paying on the whole
    balance?" -- it prices the zero-rate accrued-interest bucket correctly,
    which a weighted average of the loan rates cannot do. Bisection; returns
    None if no rate in [0, 100%] fits.
    """
    if present_value <= 0 or not payments:
        return None

    def npv(monthly_rate: float) -> float:
        return present_value - sum(
            p / (1.0 + monthly_rate) ** t for t, p in enumerate(payments, start=1)
        )

    lo, hi = 0.0, 1.0  # 0% to 100% monthly
    if npv(lo) > 0:  # payments never repay the principal
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) < 0:
            lo = mid
        else:
            hi = mid
    monthly = (lo + hi) / 2
    return (1.0 + monthly) ** 12 - 1.0


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.3f}%"


def format_summary(s: RateSummary) -> str:
    lines = [
        f"RATE SUMMARY{f'  (as of {s.as_of})' if s.as_of else ''}",
        "=" * 62,
        f"  Principal (accruing)        {money(s.principal):>18}",
        f"  Accrued interest (0% bucket){money(s.accrued_interest):>18}",
        f"  Total balance               {money(s.total_balance):>18}",
        "",
        f"  Weighted average rate       {pct(s.weighted_average_rate):>18}   <- rate you are charged, on principal",
        f"  Effective rate on balance   {pct(s.effective_rate_on_balance):>18}   <- accrual / total balance",
        f"  Unweighted mean of rates    {pct(s.simple_average_rate):>18}   (shown only to contrast; don't use it)",
        "",
        f"  Accrual                     {money(s.daily_accrual):>18} / day",
        f"                              {money(s.annual_accrual):>18} / year",
    ]
    if s.total_minimum > CENT:
        monthly_interest = s.annual_accrual / 12
        headroom = s.total_minimum - monthly_interest
        lines += [
            "",
            f"  Required minimum            {money(s.total_minimum):>18} / month",
            f"  Interest portion of that    {money(monthly_interest):>18} / month",
            f"  Principal at minimum only   {money(headroom):>18} / month",
        ]
    lines += [
        "",
        "MARGINAL RATE BY DOLLAR (avalanche order)",
        "-" * 62,
        "  What each dollar of payment actually saves you in future interest.",
        "",
        f"  {'tranche':<34}{'amount':>14}{'marginal':>10}",
    ]
    cumulative = 0.0
    for label, amount, rate in s.tranches:
        cumulative += amount
        lines.append(f"  {label:<34}{money(amount):>14}{pct(rate):>10}")
    lines += [
        "-" * 62,
        f"  {'total':<34}{money(cumulative):>14}",
        "",
        "  The first tranche is 0%: retiring accrued interest lowers your",
        "  balance but not your accrual. That is why the effective rate on the",
        "  total balance understates the rate on the dollars that matter.",
        "  It stops being 0% if the interest capitalizes -- see --capitalize-on.",
    ]
    return "\n".join(lines)


def format_sim(r: SimResult) -> str:
    lines = [
        f"SIMULATION -- {r.strategy}, {money(r.monthly_payment)}/month from {r.start}",
        "=" * 62,
    ]
    if r.payment_steps:
        lines.append("  Payment schedule:")
        lines.append(f"    {r.start.isoformat()}   {money(r.monthly_payment)}")
        for step in sorted(r.payment_steps, key=lambda s: s.effective):
            reached = r.payoff_date is None or step.effective <= r.payoff_date
            note = "" if reached else "   (loans already gone)"
            lines.append(f"    {step.effective.isoformat()}   {money(step.amount)}{note}")
        lines.append("")
    if not r.completed:
        lines.append(f"  NOT PAID OFF within {r.months} months.")
        if r.negative_amortization:
            lines.append("  Payment does not cover monthly interest: the balance is growing.")
        lines.append(f"  Balance after {r.months} months: {money(r.rows[-1]['balance_remaining'])}")
        return "\n".join(lines)

    years, months = divmod(r.months, 12)
    lines += [
        f"  Payoff date                 {r.payoff_date.isoformat():>18}",
        f"  Time to payoff              {f'{r.months} mo ({years}y {months}m)':>18}",
        f"  Starting balance            {money(r.starting_balance):>18}",
        f"  Total paid                  {money(r.total_paid):>18}",
        f"  Interest paid               {money(r.total_interest):>18}",
        f"  Realized effective APR      {pct(r.effective_apr):>18}   <- IRR on the full starting balance",
    ]
    if r.total_capitalized > CENT:
        lines.append(f"  Interest capitalized        {money(r.total_capitalized):>18}   (moved into principal, now accruing)")
    if r.loan_payoff:
        lines += ["", "  Loans cleared, in order:"]
        for name, when in sorted(r.loan_payoff.items(), key=lambda kv: kv[1]):
            lines.append(f"    {when.isoformat()}   {name}")
    if r.below_minimums:
        lines.append("  WARNING: payment is below the servicer's required minimums (delinquency).")
    if r.negative_amortization:
        lines.append("  WARNING: at least one month's payment did not cover accrued interest.")
    return "\n".join(lines)


def format_comparison(results: Sequence[SimResult], minimums_modeled: bool = False) -> str:
    baseline = next((r for r in results if r.strategy == "proportional"), results[0])
    lines = [
        "STRATEGY COMPARISON",
        "=" * 78,
        f"  {'strategy':<15}{'payoff':>12}{'months':>9}{'interest':>15}{'eff. APR':>11}{'vs base':>14}",
        "-" * 78,
    ]
    for r in results:
        if not r.completed:
            lines.append(f"  {r.strategy:<15}{'not paid off':>12}")
            continue
        delta = baseline.total_interest - r.total_interest
        delta_str = "--" if r is baseline else f"{'-' if delta > 0 else '+'}{money(abs(delta))}"
        lines.append(
            f"  {r.strategy:<15}{r.payoff_date.isoformat():>12}{r.months:>9}"
            f"{money(r.total_interest):>15}{pct(r.effective_apr):>11}{delta_str:>14}"
        )
    lines += [
        "-" * 78,
        "  'vs base' is interest saved relative to proportional (servicer default).",
        "  Effective APR is the IRR of the actual cashflows, so it prices the",
        "  zero-rate accrued-interest bucket correctly.",
    ]
    lines.append(
        "  All strategies pay the servicer's required minimum on every loan;\n"
        "  they differ only in where the excess above that goes."
        if minimums_modeled
        else "  No minimum_payment column found, so the floor is this month's interest.\n"
        "  With real minimums the strategies converge somewhat."
    )
    return "\n".join(lines)


def write_schedule(rows: Sequence[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="student_loans",
        description="Effective-rate math and payoff simulation for federal student loans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("template", help="print a starter loans.csv to stdout")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("csv", help="path to loans CSV")

    def add_sim_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--payment", type=float, required=True, help="monthly payment, dollars")
        p.add_argument(
            "--start",
            type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
            default=None,
            help="first accrual date (default: today)",
        )
        p.add_argument(
            "--rate-change",
            type=parse_rate_change,
            action="append",
            default=[],
            metavar="DATE:DELTA[:Loan|Loan]",
            help="extra rate change in percentage points, e.g. 2028-06-01:+1.0 "
            "(repeatable). Applied on top of any rate_reduction_expires column.",
        )
        p.add_argument(
            "--reduction-delta",
            type=float,
            default=1.0,
            help="percentage points the rate rises when a rate_reduction_expires "
            "date passes (default: 1.0)",
        )
        p.add_argument(
            "--no-auto-rate-change",
            action="store_true",
            help="ignore the rate_reduction_expires column",
        )
        p.add_argument(
            "--capitalize-on",
            type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
            action="append",
            default=[],
            metavar="DATE",
            help="capitalize outstanding interest on this date (repeatable)",
        )
        p.add_argument(
            "--payment-step",
            type=parse_payment_step,
            action="append",
            default=[],
            metavar="DATE:AMOUNT",
            help="raise (or lower) the monthly payment on a date, e.g. "
            "2027-04-01:2500 (repeatable)",
        )
        p.add_argument("--max-months", type=int, default=600)
        p.add_argument("--schedule-out", metavar="PATH", help="write the monthly schedule to CSV")

    p_sum = sub.add_parser("summary", help="rate math only, no simulation")
    add_common(p_sum)

    p_sim = sub.add_parser("simulate", help="run one payoff strategy")
    add_common(p_sim)
    add_sim_args(p_sim)
    p_sim.add_argument("--strategy", choices=STRATEGIES, default="avalanche")
    p_sim.add_argument(
        "--priority",
        default="",
        metavar="AH,AG,AL",
        help="comma-separated loan ids to target first, in order "
        "(implies --strategy custom; unnamed loans fall back to avalanche order)",
    )

    p_cmp = sub.add_parser("compare", help="run and compare all strategies")
    add_common(p_cmp)
    add_sim_args(p_cmp)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "template":
        sys.stdout.write(TEMPLATE)
        return 0

    try:
        loans = load_loans(args.csv)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_summary(rate_summary(loans)))

    if args.command == "summary":
        return 0

    auto = [] if args.no_auto_rate_change else auto_rate_changes(loans, args.reduction_delta)
    sim_kwargs = dict(
        start=args.start,
        rate_changes=auto + args.rate_change,
        capitalize_on=args.capitalize_on,
        max_months=args.max_months,
        payment_steps=args.payment_step,
    )
    for rc in auto:
        scope = "all loans" if rc.only is None else ", ".join(rc.only)
        print(f"\n  rate change scheduled: {rc.effective} {rc.delta_pct_points:+.2f} pp ({scope})")

    if args.command == "simulate":
        priority = tuple(n.strip() for n in args.priority.split(",") if n.strip())
        strategy = "custom" if priority else args.strategy
        result = simulate(loans, args.payment, strategy, priority=priority, **sim_kwargs)
        print()
        print(format_sim(result))
        if args.schedule_out:
            write_schedule(result.rows, args.schedule_out)
            print(f"\n  schedule written to {args.schedule_out}")
        return 0

    results = [
        simulate(loans, args.payment, s, **sim_kwargs)
        for s in STRATEGIES
        if s != "custom"
    ]
    print()
    print(format_comparison(results, minimums_modeled=any(l.minimum_payment for l in loans)))
    if args.schedule_out:
        best = min((r for r in results if r.completed), key=lambda r: r.total_interest, default=None)
        if best:
            write_schedule(best.rows, args.schedule_out)
            print(f"\n  schedule for '{best.strategy}' written to {args.schedule_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
