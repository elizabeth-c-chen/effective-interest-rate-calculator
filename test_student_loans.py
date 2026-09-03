#!/usr/bin/env python3
"""Tests for student_loans.py. Run: python -m unittest -v test_student_loans"""

import unittest
from datetime import date

from student_loans import (
    Loan,
    RateChange,
    add_months,
    irr,
    parse_rate,
    rate_summary,
    simulate,
)

START = date(2026, 9, 1)


def sample() -> list[Loan]:
    """Mixed-rate book with an accrued-interest bucket."""
    return [
        Loan("A", 40_000, 0.0628, 3_000),
        Loan("B", 50_000, 0.0453, 4_000),
        Loan("C", 40_000, 0.0373, 2_000),
    ]


class TestParsing(unittest.TestCase):
    def test_percent_and_decimal_forms_agree(self):
        for form in (5.5, "5.5", "5.5%", " 5.5 ", 0.055, "0.055"):
            self.assertAlmostEqual(parse_rate(form), 0.055, places=10)

    def test_zero(self):
        self.assertEqual(parse_rate(0), 0.0)

    def test_rejects_out_of_range_rate(self):
        with self.assertRaises(ValueError):
            Loan("bad", 1000, 1.5)  # 150% -- almost certainly a data error

    def test_add_months_clamps_day(self):
        self.assertEqual(add_months(date(2026, 1, 31), 1), date(2026, 2, 28))
        self.assertEqual(add_months(date(2028, 1, 31), 1), date(2028, 2, 29))  # leap
        self.assertEqual(add_months(date(2026, 12, 15), 1), date(2027, 1, 15))
        self.assertEqual(add_months(date(2026, 3, 31), 12), date(2027, 3, 31))


class TestRateMath(unittest.TestCase):
    def test_effective_rate_is_weighted_scaled_by_principal_share(self):
        """The whole premise: effective = weighted * principal / total."""
        s = rate_summary(sample())
        expected = s.weighted_average_rate * s.principal / s.total_balance
        self.assertAlmostEqual(s.effective_rate_on_balance, expected, places=12)
        self.assertLess(s.effective_rate_on_balance, s.weighted_average_rate)

    def test_weighted_average_by_hand(self):
        s = rate_summary(sample())
        expected = (40_000 * 0.0628 + 50_000 * 0.0453 + 40_000 * 0.0373) / 130_000
        self.assertAlmostEqual(s.weighted_average_rate, expected, places=12)

    def test_weighted_uses_principal_not_total_balance(self):
        """Weighting by balance-including-interest is a real and wrong temptation."""
        s = rate_summary(sample())
        by_balance = (43_000 * 0.0628 + 54_000 * 0.0453 + 42_000 * 0.0373) / 139_000
        self.assertNotAlmostEqual(s.weighted_average_rate, by_balance, places=5)

    def test_no_accrued_interest_means_rates_coincide(self):
        loans = [Loan("A", 40_000, 0.0628), Loan("B", 60_000, 0.04)]
        s = rate_summary(loans)
        self.assertAlmostEqual(s.effective_rate_on_balance, s.weighted_average_rate, places=12)

    def test_daily_accrual_by_hand(self):
        s = rate_summary([Loan("A", 130_000, 0.05, 9_000)])
        self.assertAlmostEqual(s.daily_accrual, 130_000 * 0.05 / 365, places=10)
        self.assertAlmostEqual(s.annual_accrual, 6_500.0, places=10)

    def test_accrued_interest_tranche_is_zero_rate_and_first(self):
        s = rate_summary(sample())
        label, amount, rate = s.tranches[0]
        self.assertIn("accrued interest", label)
        self.assertEqual(amount, 9_000)
        self.assertEqual(rate, 0.0)
        # remaining tranches are principal, in descending rate order
        rates = [r for _, _, r in s.tranches[1:]]
        self.assertEqual(rates, sorted(rates, reverse=True))
        self.assertAlmostEqual(sum(a for _, a, _ in s.tranches), s.total_balance, places=6)

    def test_simple_mean_differs_from_weighted(self):
        s = rate_summary(sample())
        self.assertNotAlmostEqual(s.simple_average_rate, s.weighted_average_rate, places=5)


class TestLoanMechanics(unittest.TestCase):
    def test_interest_does_not_accrue_on_interest(self):
        """Two loans, same total balance, different principal/interest split."""
        pure = Loan("pure", 100_000, 0.05, 0)
        split = Loan("split", 90_000, 0.05, 10_000)
        self.assertEqual(pure.balance, split.balance)
        pure.accrue(365)
        split.accrue(365)
        self.assertAlmostEqual(pure.accrued_interest, 5_000, places=6)
        self.assertAlmostEqual(split.accrued_interest, 10_000 + 4_500, places=6)

    def test_payment_hits_interest_first(self):
        loan = Loan("A", 10_000, 0.05, 300)
        interest, principal = loan.pay(500)
        self.assertAlmostEqual(interest, 300)
        self.assertAlmostEqual(principal, 200)
        self.assertAlmostEqual(loan.principal, 9_800)
        self.assertEqual(loan.accrued_interest, 0)

    def test_payment_never_overpays(self):
        loan = Loan("A", 100, 0.05, 10)
        interest, principal = loan.pay(1_000)
        self.assertAlmostEqual(interest + principal, 110)
        self.assertTrue(loan.paid_off)

    def test_capitalization_moves_interest_into_principal(self):
        loan = Loan("A", 90_000, 0.05, 10_000)
        moved = loan.capitalize()
        self.assertAlmostEqual(moved, 10_000)
        self.assertAlmostEqual(loan.principal, 100_000)
        self.assertEqual(loan.accrued_interest, 0)
        self.assertAlmostEqual(loan.balance, 100_000)  # balance unchanged...
        loan.accrue(365)
        self.assertAlmostEqual(loan.accrued_interest, 5_000)  # ...but accrual isn't


class TestSimulationInvariants(unittest.TestCase):
    def setUp(self):
        self.result = simulate(sample(), 2_500, "avalanche", start=START)

    def test_pays_off(self):
        self.assertTrue(self.result.completed)
        self.assertGreater(self.result.months, 0)

    def test_money_is_conserved(self):
        r = self.result
        accrued_during = sum(row["interest_accrued"] for row in r.rows)
        starting_accrued = 9_000
        self.assertAlmostEqual(r.total_interest, starting_accrued + accrued_during, delta=0.05)
        self.assertAlmostEqual(
            r.total_paid, r.starting_balance + accrued_during, delta=0.05
        )
        self.assertAlmostEqual(
            r.total_paid - r.total_interest, 130_000, delta=0.05  # principal retired
        )

    def test_balance_is_monotonically_decreasing(self):
        balances = [row["balance_remaining"] for row in self.result.rows]
        self.assertEqual(balances, sorted(balances, reverse=True))
        self.assertAlmostEqual(balances[-1], 0.0, places=2)

    def test_every_payment_is_the_full_amount_except_the_last(self):
        payments = [row["payment"] for row in self.result.rows]
        for p in payments[:-1]:
            self.assertAlmostEqual(p, 2_500, places=2)
        self.assertLessEqual(payments[-1], 2_500 + 0.01)

    def test_larger_payment_costs_less_interest(self):
        small = simulate(sample(), 1_500, "avalanche", start=START)
        large = simulate(sample(), 3_500, "avalanche", start=START)
        self.assertLess(large.total_interest, small.total_interest)
        self.assertLess(large.months, small.months)

    def test_negative_amortization_flagged_and_no_false_positive(self):
        """$130k at ~4.9% accrues ~$530/mo; $300/mo can't keep up."""
        starving = simulate(sample(), 300, "avalanche", start=START, max_months=24)
        self.assertTrue(starving.negative_amortization)
        self.assertFalse(starving.completed)
        self.assertGreater(starving.rows[-1]["balance_remaining"], 139_000 - 300 * 24)
        self.assertFalse(self.result.negative_amortization)


class TestStrategies(unittest.TestCase):
    def test_avalanche_beats_or_ties_proportional(self):
        for payment in (1_500, 2_000, 2_500, 3_500, 5_000):
            with self.subTest(payment=payment):
                av = simulate(sample(), payment, "avalanche", start=START)
                pr = simulate(sample(), payment, "proportional", start=START)
                self.assertLessEqual(av.total_interest, pr.total_interest + 0.01)
                self.assertLessEqual(av.months, pr.months)

    def test_avalanche_beats_or_ties_snowball_on_interest(self):
        av = simulate(sample(), 2_500, "avalanche", start=START)
        sn = simulate(sample(), 2_500, "snowball", start=START)
        self.assertLessEqual(av.total_interest, sn.total_interest + 0.01)

    def test_all_strategies_retire_the_same_principal(self):
        for strategy in ("avalanche", "proportional", "snowball"):
            with self.subTest(strategy=strategy):
                r = simulate(sample(), 2_500, strategy, start=START)
                self.assertAlmostEqual(r.total_paid - r.total_interest, 130_000, delta=0.05)

    def test_strategy_is_irrelevant_when_rates_are_equal_and_no_backlog(self):
        flat = [Loan("A", 40_000, 0.05), Loan("B", 60_000, 0.05)]
        av = simulate(flat, 2_000, "avalanche", start=START)
        pr = simulate(flat, 2_000, "proportional", start=START)
        self.assertAlmostEqual(av.total_interest, pr.total_interest, delta=1.0)

    def test_avalanche_wins_even_at_equal_rates_when_a_backlog_exists(self):
        """Not obvious: with identical rates, ordering still matters, because
        the accrued-interest backlog is a 0% tranche. Avalanche pays only this
        month's interest and puts the rest into principal; the proportional
        default drains the 0% backlog first, which saves nothing."""
        flat = [Loan("A", 40_000, 0.05, 1_000), Loan("B", 60_000, 0.05, 2_000)]
        av = simulate(flat, 2_000, "avalanche", start=START)
        pr = simulate(flat, 2_000, "proportional", start=START)
        self.assertLess(av.total_interest, pr.total_interest)

    def test_simulation_does_not_mutate_input(self):
        loans = sample()
        before = [(l.principal, l.rate, l.accrued_interest) for l in loans]
        simulate(loans, 2_500, "avalanche", start=START)
        after = [(l.principal, l.rate, l.accrued_interest) for l in loans]
        self.assertEqual(before, after)


class TestEffectiveAPR(unittest.TestCase):
    def test_irr_recovers_the_nominal_rate_when_there_is_no_interest_bucket(self):
        """Single loan, zero accrued interest: the IRR must be the loan's rate.

        Reported as an effective annual rate, so the target is (1+r/12)^12-1.
        """
        for rate in (0.0373, 0.05, 0.0628):
            with self.subTest(rate=rate):
                r = simulate([Loan("A", 100_000, rate)], 1_500, "avalanche", start=START)
                target = (1 + rate / 12) ** 12 - 1
                self.assertAlmostEqual(r.effective_apr, target, delta=0.001)

    def test_irr_is_below_weighted_average_when_interest_bucket_exists(self):
        """The question that started this: the zero-rate bucket drags it down."""
        loans = sample()
        s = rate_summary(loans)
        r = simulate(loans, 2_500, "avalanche", start=START)
        self.assertLess(r.effective_apr, s.weighted_average_rate)

    def test_irr_closed_form_against_a_known_annuity(self):
        """Price a textbook annuity: borrow 10k, repay 12 x 900."""
        rate = irr(10_000, [900.0] * 12)
        monthly = (1 + rate) ** (1 / 12) - 1
        pv = sum(900 / (1 + monthly) ** t for t in range(1, 13))
        self.assertAlmostEqual(pv, 10_000, places=4)

    def test_irr_returns_none_when_payments_cannot_repay(self):
        self.assertIsNone(irr(10_000, [10.0] * 12))


class TestRateChangesAndCapitalization(unittest.TestCase):
    def test_losing_an_autopay_discount_costs_money(self):
        base = simulate(sample(), 2_000, "avalanche", start=START)
        worse = simulate(
            sample(),
            2_000,
            "avalanche",
            start=START,
            rate_changes=[RateChange(date(2028, 6, 1), +1.0)],
        )
        self.assertGreater(worse.total_interest, base.total_interest)
        self.assertGreaterEqual(worse.months, base.months)

    def test_rate_change_before_payoff_only(self):
        """A change scheduled after payoff must not affect anything."""
        base = simulate(sample(), 5_000, "avalanche", start=START)
        later = simulate(
            sample(),
            5_000,
            "avalanche",
            start=START,
            rate_changes=[RateChange(date(2099, 1, 1), +5.0)],
        )
        self.assertAlmostEqual(base.total_interest, later.total_interest, places=6)

    def test_rate_change_can_target_specific_loans(self):
        all_loans = simulate(
            sample(), 2_000, "avalanche", start=START,
            rate_changes=[RateChange(date(2027, 1, 1), +1.0)],
        )
        one_loan = simulate(
            sample(), 2_000, "avalanche", start=START,
            rate_changes=[RateChange(date(2027, 1, 1), +1.0, only=("A",))],
        )
        self.assertLess(one_loan.total_interest, all_loans.total_interest)

    def test_capitalization_makes_the_zero_rate_bucket_cost_money(self):
        base = simulate(sample(), 2_000, "avalanche", start=START)
        capped = simulate(
            sample(), 2_000, "avalanche", start=START, capitalize_on=[date(2026, 10, 1)]
        )
        self.assertGreater(capped.total_interest, base.total_interest)

    def test_capitalization_is_harmless_with_nothing_to_capitalize(self):
        clean = [Loan("A", 40_000, 0.0628), Loan("B", 60_000, 0.04)]
        base = simulate(clean, 2_500, "proportional", start=START)
        capped = simulate(
            clean, 2_500, "proportional", start=START, capitalize_on=[date(2026, 10, 1)]
        )
        # Capitalizing one month's accrual on day one is a rounding-level event.
        self.assertAlmostEqual(base.total_interest, capped.total_interest, delta=25.0)
        self.assertGreaterEqual(capped.total_interest, base.total_interest)


class TestMinimums(unittest.TestCase):
    """Minimums are a constraint, not a strategy: they bind every strategy equally."""

    def book(self):
        return [
            Loan("hi", 20_000, 0.065, 1_500, minimum_payment=140),
            Loan("lo", 20_000, 0.033, 1_500, minimum_payment=140),
        ]

    def test_minimums_are_always_paid(self):
        r = simulate(self.book(), 1_000, "avalanche", start=START)
        self.assertFalse(r.below_minimums)
        # every non-final month clears at least the required 280
        for row in r.rows[:-1]:
            self.assertGreaterEqual(row["payment"], 280 - 0.01)

    def test_paying_under_the_minimum_is_flagged(self):
        r = simulate(self.book(), 200, "avalanche", start=START, max_months=12)
        self.assertTrue(r.below_minimums)

    def test_minimums_narrow_the_gap_between_strategies(self):
        """The whole point of collecting the column: without minimums avalanche
        can hoard the 0% backlog; with them, money is forced into every loan."""
        free = [Loan("hi", 20_000, 0.065, 1_500), Loan("lo", 20_000, 0.033, 1_500)]
        bound = self.book()
        gap_free = (
            simulate(free, 1_000, "proportional", start=START).total_interest
            - simulate(free, 1_000, "avalanche", start=START).total_interest
        )
        gap_bound = (
            simulate(bound, 1_000, "proportional", start=START).total_interest
            - simulate(bound, 1_000, "avalanche", start=START).total_interest
        )
        self.assertGreater(gap_free, gap_bound)
        self.assertGreater(gap_bound, 0)  # avalanche still wins, just by less

    def test_minimum_is_capped_at_payoff_amount(self):
        """A loan with $10 left must not demand its full $140 minimum."""
        book = [Loan("tiny", 5, 0.05, 0, minimum_payment=140), Loan("big", 20_000, 0.05, 0, minimum_payment=140)]
        r = simulate(book, 500, "avalanche", start=START)
        self.assertTrue(r.completed)
        self.assertAlmostEqual(r.total_paid - r.total_interest, 20_005, delta=0.05)


class TestCustomPriority(unittest.TestCase):
    def book(self):
        return [
            Loan("small", 2_000, 0.055, 10, minimum_payment=27),
            Loan("hi", 20_000, 0.065, 1_500, minimum_payment=140),
            Loan("lo", 20_000, 0.033, 1_500, minimum_payment=140),
        ]

    def test_priority_loan_clears_first(self):
        r = simulate(self.book(), 1_000, "custom", start=START, priority=("small",))
        first = min(r.loan_payoff.items(), key=lambda kv: kv[1])[0]
        self.assertEqual(first, "small")

    def test_avalanche_clears_highest_rate_first(self):
        r = simulate(self.book(), 1_000, "avalanche", start=START)
        first = min(r.loan_payoff.items(), key=lambda kv: kv[1])[0]
        self.assertEqual(first, "hi")

    def test_every_loan_gets_a_payoff_date(self):
        r = simulate(self.book(), 1_000, "avalanche", start=START)
        self.assertEqual(set(r.loan_payoff), {"small", "hi", "lo"})
        self.assertEqual(max(r.loan_payoff.values()), r.payoff_date)

    def test_custom_costs_something_but_never_beats_avalanche(self):
        av = simulate(self.book(), 1_000, "avalanche", start=START)
        cu = simulate(self.book(), 1_000, "custom", start=START, priority=("small",))
        self.assertGreaterEqual(cu.total_interest, av.total_interest - 0.01)

    def test_empty_priority_falls_back_to_avalanche_order(self):
        av = simulate(self.book(), 1_000, "avalanche", start=START)
        cu = simulate(self.book(), 1_000, "custom", start=START, priority=())
        self.assertAlmostEqual(av.total_interest, cu.total_interest, places=6)

    def test_unknown_priority_name_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate(self.book(), 1_000, "custom", start=START, priority=("typo",))


class TestPaymentSteps(unittest.TestCase):
    def setUp(self):
        from student_loans import PaymentStep

        self.Step = PaymentStep
        self.steps = [PaymentStep(date(2027, 4, 1), 2_500), PaymentStep(date(2028, 4, 1), 3_200)]

    def test_payment_on_picks_the_latest_step_in_force(self):
        from student_loans import payment_on

        self.assertEqual(payment_on(2_000, self.steps, date(2026, 12, 1)), 2_000)
        self.assertEqual(payment_on(2_000, self.steps, date(2027, 3, 31)), 2_000)
        self.assertEqual(payment_on(2_000, self.steps, date(2027, 4, 1)), 2_500)
        self.assertEqual(payment_on(2_000, self.steps, date(2028, 5, 1)), 3_200)

    def test_steps_are_order_independent(self):
        from student_loans import payment_on

        self.assertEqual(
            payment_on(2_000, list(reversed(self.steps)), date(2028, 5, 1)),
            payment_on(2_000, self.steps, date(2028, 5, 1)),
        )

    def test_stepping_up_beats_staying_flat(self):
        flat = simulate(sample(), 2_000, "avalanche", start=START)
        stepped = simulate(sample(), 2_000, "avalanche", start=START, payment_steps=self.steps)
        self.assertLess(stepped.months, flat.months)
        self.assertLess(stepped.total_interest, flat.total_interest)

    def test_payments_actually_follow_the_schedule(self):
        r = simulate(sample(), 2_000, "avalanche", start=START, payment_steps=self.steps)
        for row in r.rows[:-1]:
            when = date.fromisoformat(row["date"])
            expected = 2_000 if when < date(2027, 4, 1) else (2_500 if when < date(2028, 4, 1) else 3_200)
            self.assertAlmostEqual(row["payment"], expected, places=2)

    def test_no_steps_matches_a_flat_payment(self):
        flat = simulate(sample(), 2_000, "avalanche", start=START)
        empty = simulate(sample(), 2_000, "avalanche", start=START, payment_steps=[])
        self.assertAlmostEqual(flat.total_interest, empty.total_interest, places=6)

    def test_money_is_still_conserved_with_steps(self):
        r = simulate(sample(), 2_000, "avalanche", start=START, payment_steps=self.steps)
        self.assertAlmostEqual(r.total_paid - r.total_interest, 130_000, delta=0.05)


class TestFileLoading(unittest.TestCase):
    def test_servicer_column_names_load(self):
        import tempfile, os
        from student_loans import load_loans, auto_rate_changes

        csv_text = (
            "loan_id,as_of_date,principal_balance,accrued_interest,interest_rate,"
            "minimum_payment,rate_reduction_expires\n"
            "AA,2026-09-03,20758,1142.44,0.033,140.24,2028-06-01\n"
            "AG,2026-09-03,16119.49,1120.63,0.0654,108.92,2028-06-01\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write(csv_text)
            path = fh.name
        try:
            loans = load_loans(path)
            self.assertEqual([l.name for l in loans], ["AA", "AG"])
            self.assertAlmostEqual(loans[0].principal, 20758)
            self.assertAlmostEqual(loans[0].rate, 0.033)
            self.assertAlmostEqual(loans[1].minimum_payment, 108.92)
            self.assertEqual(loans[0].as_of, date(2026, 9, 3))

            changes = auto_rate_changes(loans)
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0].effective, date(2028, 6, 1))
            self.assertIsNone(changes[0].only)  # applies to everyone

            # start defaults to the as-of date, not today
            r = simulate(loans, 1_000, "avalanche")
            self.assertEqual(r.start, date(2026, 9, 3))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
