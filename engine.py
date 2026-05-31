"""Settlement feasibility & fee scheduling engine.

evaluate_offer(client, offer, rules) -> Result

Design decisions (documented here per ASSIGNMENT.md request):

PAYMENT SHAPES
--------------
even_pays=True  -> "even": all payments equal; remainder cents go to the LAST
                   payments (keeps sequence non-decreasing). We try every k from
                   1..max_k and pick the k whose feasible schedule best front-loads
                   the program fee (i.e. leaves the smallest balance after fee
                   collection).

is_ballooning_allowed=True (and not even) -> "balloon": all early payments are at
                   the effective floor for that position (min_payment_cents, respecting
                   token-pay budget and tier floors), final payment absorbs remainder.
                   We try every k from 1..max_k.

Neither -> "staircase": payments use at most max_segments distinct levels, are
           non-decreasing, and satisfy all floor constraints. We greedily keep early
           payments as low as possible (to front-load the fee) and raise later ones.
           Specifically: we try all valid (k, segment-boundary) combinations, simulate
           each, and pick the feasible one that maximises total fee collected in the
           earliest slots.

TOKEN PAYS + TIERS
------------------
floor(i) = max(min_payment_cents,
               min_payment_cents + epsilon if token budget exhausted,
               tier_floor(i))
where tier_floor(i) is from min_payment_tiers for 1-based index i.
A payment at exactly min_payment_cents counts as a token pay. Once max_token_pays
token pays have been used, every subsequent payment must STRICTLY exceed
min_payment_cents (we use min_payment_cents+1 as the effective floor in that case).

FEE SCHEDULING
--------------
On each cadence date we greedily collect as much program fee as possible:
    fee_this_date = min(remaining_program_fee, balance_after_creditor_and_bank_fees)
This is the front-loading objective. A fee-only cadence date (creditor_payment=0) is
allowed after all creditor payments are done, to mop up any remaining fee, as long as
it is <= horizon.

SIMULATION
----------
We merge all dated events (ledger entries + creditor payments + fees) and process
them in date order, applying credits before debits on the same date.

INFEASIBILITY
-------------
If no valid schedule exists for any k, we compute:
  - Minimum lump sum L (on first_draft_date, the earliest useful date) via binary
    search, then check guardrail L <= round(0.65 * offer_total).
  - Minimum per-draft increment X (across all N future drafts) via binary search,
    then check guardrail X <= max(10000, round(0.40 * draft_amount_cents)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

from feasibility.models import (
    Client,
    CreditorRules,
    LedgerEntry,
    Offer,
    add_months,
    default_first_payment_date,
    end_of_month,
    is_end_of_month,
    monthly_payment_dates,
    offer_total_cents,
    program_fee_cents,
)


# ---------------------------------------------------------------------------
# Output dataclasses (keep in engine.py so runner/tests import them correctly)
# ---------------------------------------------------------------------------

@dataclass
class ScheduleRow:
    date: date
    creditor_payment_cents: int
    program_fee_cents: int
    bank_fee_cents: int
    balance_cents: int


@dataclass
class FundsOption:
    amount_cents: int
    within_guardrail: bool
    reason: str
    date: date | None = None
    num_drafts: int | None = None


@dataclass
class AdditionalFunds:
    lump_sum: FundsOption
    monthly_increment: FundsOption


@dataclass
class Result:
    feasible: bool
    pay_shape_used: str | None = None
    schedule: list[ScheduleRow] | None = None
    additional_funds: AdditionalFunds | None = None

    def to_dict(self) -> dict:
        out: dict = {"feasible": self.feasible, "pay_shape_used": self.pay_shape_used}
        out["schedule"] = (
            [
                {
                    "date": r.date.isoformat(),
                    "creditor_payment_cents": r.creditor_payment_cents,
                    "program_fee_cents": r.program_fee_cents,
                    "bank_fee_cents": r.bank_fee_cents,
                    "balance_cents": r.balance_cents,
                }
                for r in self.schedule
            ]
            if self.schedule is not None
            else None
        )
        if self.additional_funds is None:
            out["additional_funds"] = None
        else:
            def opt(o: FundsOption) -> dict:
                d = {
                    "amount_cents": o.amount_cents,
                    "within_guardrail": o.within_guardrail,
                    "reason": o.reason,
                }
                if o.date is not None:
                    d["date"] = o.date.isoformat()
                if o.num_drafts is not None:
                    d["num_drafts"] = o.num_drafts
                return d

            out["additional_funds"] = {
                "lump_sum": opt(self.additional_funds.lump_sum),
                "monthly_increment": opt(self.additional_funds.monthly_increment),
            }
        return out


# ---------------------------------------------------------------------------
# Rounding (round-half-up, not Python's default banker's rounding)
# ---------------------------------------------------------------------------

def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


# ---------------------------------------------------------------------------
# Floor computation
# ---------------------------------------------------------------------------

def compute_floor(position_1based: int, token_pays_used: int, rules: CreditorRules) -> int:
    """Return the minimum allowed payment at this 1-based position."""
    base = rules.min_payment_cents

    # Tier floors
    tier_floor = base
    for (from_pos, min_cents) in rules.min_payment_tiers:
        if position_1based >= from_pos:
            tier_floor = max(tier_floor, min_cents)

    # Token-pay budget: if we've already used max_token_pays at base min, must exceed
    if token_pays_used >= rules.max_token_pays:
        effective_min = max(tier_floor, base + 1)
    else:
        effective_min = tier_floor

    return effective_min


# ---------------------------------------------------------------------------
# Build candidate payment schedules
# ---------------------------------------------------------------------------

def build_even_payments(k: int, offer_total: int, rules: CreditorRules) -> list[int] | None:
    """Build k equal payments summing to offer_total. Returns None if invalid."""
    if k <= 0:
        return None
    base = offer_total // k
    remainder = offer_total % k
    # remainder goes to the LAST 'remainder' payments (non-decreasing)
    payments = [base] * (k - remainder) + [base + 1] * remainder

    # Validate floors
    token_pays_used = 0
    for i, p in enumerate(payments):
        floor = compute_floor(i + 1, token_pays_used, rules)
        if p < floor:
            return None
        if p == rules.min_payment_cents:
            token_pays_used += 1

    return payments


def build_balloon_payments(k: int, offer_total: int, rules: CreditorRules) -> list[int] | None:
    """Build balloon: min payments early, final absorbs remainder."""
    if k <= 0:
        return None

    payments = []
    token_pays_used = 0
    running = 0

    for i in range(k - 1):
        floor = compute_floor(i + 1, token_pays_used, rules)
        p = floor
        payments.append(p)
        running += p
        if p == rules.min_payment_cents:
            token_pays_used += 1

    # Final payment
    final = offer_total - running
    final_floor = compute_floor(k, token_pays_used, rules)
    if final < final_floor:
        return None
    if final < payments[-1] if payments else False:
        return None  # non-decreasing violated
    payments.append(final)

    if sum(payments) != offer_total:
        return None

    return payments


def build_staircase_payments(k: int, offer_total: int, rules: CreditorRules) -> list[list[int]]:
    """
    Generate candidate staircase schedules for k payments using at most max_segments
    distinct levels. Returns a list of valid payment lists (may be empty).

    Strategy: greedily keep early payments as low as possible.
    We enumerate split points for the segments. For max_segments=S and k payments,
    we choose S-1 boundary indices where the level steps up.
    """
    max_seg = rules.max_segments
    if max_seg <= 0:
        max_seg = 1

    results = []

    def _enumerate_segments(seg_count: int):
        """Enumerate all ways to split k payments into seg_count segments."""
        if seg_count == 1:
            yield [k]
            return
        for first_seg_len in range(1, k - seg_count + 2):
            for rest in _enumerate_segments(seg_count - 1):
                yield [first_seg_len] + rest

    for seg_count in range(1, min(max_seg, k) + 1):
        for segment_lengths in _enumerate_segments(seg_count):
            if sum(segment_lengths) != k:
                continue
            payments = _build_from_segments(segment_lengths, offer_total, rules)
            if payments is not None:
                results.append(payments)

    return results


def _build_from_segments(segment_lengths: list[int], offer_total: int,
                          rules: CreditorRules) -> list[int] | None:
    """
    Build a payment schedule given segment lengths. Within each segment, all payments
    are equal (the level for that segment). Levels are non-decreasing across segments.
    Objective: keep early levels as low as possible.
    """
    k = sum(segment_lengths)
    num_segs = len(segment_lengths)

    # Determine the floor for each position
    token_pays_used = 0
    floors = []
    for i in range(k):
        f = compute_floor(i + 1, token_pays_used, rules)
        floors.append(f)
        # Tentatively assume payment is at floor to count token pays
        if f == rules.min_payment_cents:
            token_pays_used += 1

    # For each segment, the effective floor is max of all floors in that segment
    seg_floors = []
    pos = 0
    for length in segment_lengths:
        seg_floor = max(floors[pos:pos + length])
        seg_floors.append(seg_floor)
        pos += length

    # Levels must be non-decreasing; enforce each level >= previous level
    for i in range(1, num_segs):
        seg_floors[i] = max(seg_floors[i], seg_floors[i - 1])

    # Assign levels greedily: first seg_count-1 segments get their floor,
    # last segment absorbs remainder.
    levels = list(seg_floors)

    # Compute sum if all segments are at their floor level
    partial_sum = sum(levels[s] * segment_lengths[s] for s in range(num_segs - 1))
    last_seg_total = offer_total - partial_sum
    last_seg_len = segment_lengths[-1]

    if last_seg_len <= 0:
        return None

    # Last segment level must be >= floor and make the total exact
    if last_seg_total % last_seg_len != 0:
        # Last segment payments won't all be equal — distribute remainder to last payments
        last_level_base = last_seg_total // last_seg_len
        last_remainder = last_seg_total % last_seg_len
        if last_level_base < levels[-1]:
            return None
    else:
        last_level_base = last_seg_total // last_seg_len
        last_remainder = 0
        if last_level_base < levels[-1]:
            return None

    levels[-1] = last_level_base

    # Build actual payment list
    payments = []
    token_pays_used = 0
    for s, length in enumerate(segment_lengths):
        base_pay = levels[s]
        for j in range(length):
            pos_1based = sum(segment_lengths[:s]) + j + 1
            floor = compute_floor(pos_1based, token_pays_used, rules)

            # For last segment, distribute remainder to last payments
            if s == num_segs - 1 and j >= length - last_remainder:
                p = base_pay + 1
            else:
                p = base_pay

            if p < floor:
                return None

            payments.append(p)
            if p == rules.min_payment_cents:
                token_pays_used += 1

    if sum(payments) != offer_total:
        return None

    # Verify non-decreasing
    for i in range(1, len(payments)):
        if payments[i] < payments[i - 1]:
            return None

    return payments


# ---------------------------------------------------------------------------
# Ledger simulation
# ---------------------------------------------------------------------------

def simulate(
    client: Client,
    base_ledger: list[LedgerEntry],
    pay_dates: list[date],
    creditor_payments: list[int],
    bank_fee_per_pay: int,
    total_program_fee: int,
    first_payment_date: date,
    horizon: date,
    extra_lump: int = 0,
    lump_date: date | None = None,
    extra_per_draft: int = 0,
) -> tuple[bool, list[ScheduleRow]]:
    """
    Simulate the SDA ledger. Returns (feasible, schedule_rows).

    Schedule rows cover only cadence dates (creditor payment dates and fee-only dates).
    Credits are applied before debits on each date.
    Program fee is collected greedily (front-loaded) on each cadence date.
    """
    # Build a map: date -> net credit from ledger (only entries after as_of_date)
    credit_map: dict[date, int] = {}
    for entry in base_ledger:
        if entry.date <= client.as_of_date:
            continue
        if entry.type == "credit":
            credit_map[entry.date] = credit_map.get(entry.date, 0) + entry.amount_cents
            if extra_per_draft > 0 and entry.amount_cents == client.draft_amount_cents:
                credit_map[entry.date] = credit_map.get(entry.date, 0) + extra_per_draft
        else:  # debit (fixed, don't touch)
            credit_map[entry.date] = credit_map.get(entry.date, 0) - entry.amount_cents

    if extra_lump and lump_date:
        credit_map[lump_date] = credit_map.get(lump_date, 0) + extra_lump

    # Collect all dates we need to process: ledger dates + pay_dates
    pay_date_set = set(pay_dates)
    all_dates = sorted(set(list(credit_map.keys()) + list(pay_dates)))

    balance = client.current_balance_cents
    prog_remaining = total_program_fee
    schedule: list[ScheduleRow] = []

    pay_index = 0  # index into pay_dates / creditor_payments

    for d in all_dates:
        if d > horizon:
            break

        # Apply ledger credits/debits for this date
        balance += credit_map.get(d, 0)

        # If this is a cadence date (creditor payment or potential fee-only)
        if d in pay_date_set:
            # Creditor payment for this date
            cp = creditor_payments[pay_index] if pay_index < len(creditor_payments) else 0
            bank_fee = bank_fee_per_pay if cp > 0 else 0

            # Deduct creditor payment + bank fee first
            balance -= cp + bank_fee
            if balance < 0:
                return False, []

            # Now greedily collect as much program fee as possible
            pf = 0
            if d >= first_payment_date and prog_remaining > 0:
                pf = min(prog_remaining, balance)
                balance -= pf
                prog_remaining -= pf

            if balance < 0:
                return False, []

            schedule.append(ScheduleRow(
                date=d,
                creditor_payment_cents=cp,
                program_fee_cents=pf,
                bank_fee_cents=bank_fee,
                balance_cents=balance,
            ))
            pay_index += 1

    # If program fee not fully collected, try fee-only dates after last creditor payment
    if prog_remaining > 0:
        # Get the next cadence date after last pay_date
        last_pay_date = pay_dates[-1] if pay_dates else first_payment_date
        fee_date = add_months(last_pay_date, 1)
        # True EOM cadence
        if is_end_of_month(last_pay_date):
            fee_date = end_of_month(fee_date)

        while prog_remaining > 0 and fee_date <= horizon:
            # Apply any credits on this date
            balance += credit_map.get(fee_date, 0)
            pf = min(prog_remaining, balance)
            balance -= pf
            prog_remaining -= pf
            if balance < 0:
                return False, []
            schedule.append(ScheduleRow(
                date=fee_date,
                creditor_payment_cents=0,
                program_fee_cents=pf,
                bank_fee_cents=0,
                balance_cents=balance,
            ))
            last_pay_date = fee_date
            fee_date = add_months(fee_date, 1)
            if is_end_of_month(last_pay_date):
                fee_date = end_of_month(fee_date)

    # Feasible only if fee fully collected and balance never went negative
    return prog_remaining == 0, schedule


# ---------------------------------------------------------------------------
# Core: try all candidate schedules for a given k
# ---------------------------------------------------------------------------

def _max_k(client: Client, rules: CreditorRules, first_payment_date: date) -> int:
    horizon = client.last_draft_date
    max_by_rules = min(rules.max_payments, rules.max_terms)
    # Count how many cadence dates fit within horizon
    k = 0
    while True:
        d = add_months(first_payment_date, k)
        if is_end_of_month(first_payment_date):
            d = end_of_month(d)
        if d > horizon:
            break
        k += 1
        if k >= max_by_rules:
            break
    return min(k, max_by_rules)


def _score_schedule(schedule: list[ScheduleRow]) -> int:
    """Score = sum of (prog_fee * weight) where earlier dates get higher weight.
    Higher score = fee collected earlier = better."""
    score = 0
    for i, row in enumerate(schedule):
        weight = len(schedule) - i
        score += row.program_fee_cents * weight
    return score


def _try_feasible(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
    shape: str,
    extra_lump: int = 0,
    lump_date: date | None = None,
    extra_per_draft: int = 0,
) -> tuple[bool, list[ScheduleRow], str]:
    """Try to find a feasible schedule. Returns (feasible, best_schedule, shape_used)."""

    fpd = offer.first_payment_date or default_first_payment_date(client)
    horizon = client.last_draft_date
    offer_total = offer_total_cents(offer)
    prog_fee = program_fee_cents(offer, rules)
    max_k = _max_k(client, rules, fpd)

    best_schedule: list[ScheduleRow] = []
    best_score = -1
    found = False

    for k in range(1, max_k + 1):
        pay_dates = monthly_payment_dates(fpd, k)
        if not pay_dates or pay_dates[-1] > horizon:
            continue

        # Generate candidate payment lists based on shape
        candidates: list[list[int]] = []

        if shape == "even":
            p = build_even_payments(k, offer_total, rules)
            if p:
                candidates = [p]
        elif shape == "balloon":
            p = build_balloon_payments(k, offer_total, rules)
            if p:
                candidates = [p]
        else:  # staircase
            candidates = build_staircase_payments(k, offer_total, rules)

        for payments in candidates:
            ok, schedule = simulate(
                client=client,
                base_ledger=client.ledger,
                pay_dates=pay_dates,
                creditor_payments=payments,
                bank_fee_per_pay=rules.bank_fee_cents,
                total_program_fee=prog_fee,
                first_payment_date=fpd,
                horizon=horizon,
                extra_lump=extra_lump,
                lump_date=lump_date,
                extra_per_draft=extra_per_draft,
            )
            if ok:
                score = _score_schedule(schedule)
                if score > best_score:
                    best_score = score
                    best_schedule = schedule
                    found = True

    return found, best_schedule, shape


# ---------------------------------------------------------------------------
# Part 2: minimum additional funds
# ---------------------------------------------------------------------------

def _find_min_lump(client: Client, offer: Offer, rules: CreditorRules, shape: str) -> tuple[int, date]:
    """Binary search for minimum lump sum on first_draft_date."""
    lump_date = client.first_draft_date
    lo, hi = 0, offer_total_cents(offer) + program_fee_cents(offer, rules) + 1000000

    # Find upper bound that works
    ok, _, _ = _try_feasible(client, offer, rules, shape, extra_lump=hi, lump_date=lump_date)
    if not ok:
        return hi, lump_date  # pathological case

    while lo < hi:
        mid = (lo + hi) // 2
        ok, _, _ = _try_feasible(client, offer, rules, shape, extra_lump=mid, lump_date=lump_date)
        if ok:
            hi = mid
        else:
            lo = mid + 1

    return lo, lump_date


def _find_min_increment(client: Client, offer: Offer, rules: CreditorRules, shape: str) -> tuple[int, int]:
    """Binary search for minimum per-draft increment."""
    future_drafts = [e for e in client.ledger if e.date > client.as_of_date and e.type == "credit"]
    n = len(future_drafts)
    if n == 0:
        return 0, 0

    lo, hi = 0, offer_total_cents(offer) + program_fee_cents(offer, rules) + 1000000

    ok, _, _ = _try_feasible(client, offer, rules, shape, extra_per_draft=hi)
    if not ok:
        return hi, n

    while lo < hi:
        mid = (lo + hi) // 2
        ok, _, _ = _try_feasible(client, offer, rules, shape, extra_per_draft=mid)
        if ok:
            hi = mid
        else:
            lo = mid + 1

    return lo, n


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_offer(client: Client, offer: Offer, rules: CreditorRules) -> Result:
    """Evaluate a single offer. See ASSIGNMENT.md for the full specification."""

    # Determine payment shape
    if rules.even_pays:
        shape = "even"
    elif rules.is_ballooning_allowed:
        shape = "balloon"
    else:
        shape = "staircase"

    # Part 1: try to find a feasible schedule
    feasible, schedule, shape_used = _try_feasible(client, offer, rules, shape)

    if feasible:
        return Result(
            feasible=True,
            pay_shape_used=shape_used,
            schedule=schedule,
            additional_funds=None,
        )

    # Part 2: compute minimum additional funds
    offer_total = offer_total_cents(offer)

    # Minimum lump sum
    lump_amount, lump_date = _find_min_lump(client, offer, rules, shape)
    lump_guardrail = round_half_up(0.65 * offer_total)
    lump_within = lump_amount <= lump_guardrail
    lump_reason = "" if lump_within else (
        f"Lump sum {lump_amount} exceeds guardrail of {lump_guardrail} "
        f"(65% of offer_total {offer_total})"
    )

    # Minimum monthly increment
    inc_amount, num_drafts = _find_min_increment(client, offer, rules, shape)
    inc_guardrail = max(10000, round_half_up(0.40 * client.draft_amount_cents))
    inc_within = inc_amount <= inc_guardrail
    inc_reason = "" if inc_within else (
        f"Increment {inc_amount} exceeds guardrail of {inc_guardrail} "
        f"(max(10000, 40% of draft {client.draft_amount_cents}))"
    )

    return Result(
        feasible=False,
        pay_shape_used=None,
        schedule=None,
        additional_funds=AdditionalFunds(
            lump_sum=FundsOption(
                amount_cents=lump_amount,
                within_guardrail=lump_within,
                reason=lump_reason,
                date=lump_date,
            ),
            monthly_increment=FundsOption(
                amount_cents=inc_amount,
                within_guardrail=inc_within,
                reason=inc_reason,
                num_drafts=num_drafts,
            ),
        ),
    )
