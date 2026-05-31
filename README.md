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
