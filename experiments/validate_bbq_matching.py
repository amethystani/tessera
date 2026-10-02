"""Check BBQ group-matching coverage per category before running any model.

The decomposition needs to know which of the two substantive answers BBQ counts
as stereotype-aligned. That comes from matching an item's answer_info labels
against its stereotyped_groups annotation. If the matcher fails on a category,
the numbers for it just go missing, and a summary over "all categories" would
quietly mean the subset the matcher handled. This prints coverage per category.

Three bugs it found while the matcher was being written:
  1. SES at 0%: canon() fell back to the raw string on a dictionary miss, so
     the annotation "low SES" never equalled the answer label "lowSES".
  2. Nationality at 0%: the nationality is in answer_info field 0 and field 1
     holds only a coarse region (["British", "Europe"]), while
     stereotyped_groups names the nationality.
  3. Age at 84.8%: merging both fields fixed (2) but made "85-year-old" match
     "old", so both answers looked stereotype-aligned. The matcher now tries
     field 1 first and falls back to field 0 only if that gives no clean split.

After the fixes 9 of 11 categories are at about 100%. Race_x_SES and
Race_x_gender sit near 66% because in a third of their items both non-unknown
answers share the annotated attribute and differ on the other axis (for example
stereotyped_groups=["Black"] with answers "F-Black" and "M-Black"). There is no
target/non-target contrast there, so skipping them is correct.

Run: python3 validate_bbq_matching.py
Exits 0 if every category meets its expected floor.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bbq_firstparty import biased_index, unknown_index  # noqa: E402

DATA = Path(__file__).parents[1] / "data" / "bbq"

# Minimum acceptable usable fraction per category. The two Race_x_* categories
# have a lower floor for the substantive reason documented above, not because
# the matcher is weak there.
FLOORS = {
    "Race_x_SES": 0.60,
    "Race_x_gender": 0.60,
}
DEFAULT_FLOOR = 0.95


def main() -> None:
    failures = []
    grand_tot = grand_ok = 0
    print(f"{'category':22s} {'ambig':>6s} {'usable':>7s} {'pct':>7s}")
    for path in sorted(DATA.glob("*.jsonl")):
        tot = ok = 0
        unmapped = Counter()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["context_condition"] != "ambig":
                continue
            tot += 1
            unk = unknown_index(rec)
            if unk is not None and biased_index(rec, unk) is not None:
                ok += 1
            elif unk is not None:
                others = [i for i in range(3) if i != unk]
                unmapped[(
                    tuple(sorted(str(g) for g in
                                 rec["additional_metadata"]["stereotyped_groups"]))[:2],
                    tuple(sorted(str(rec["answer_info"][f"ans{i}"][1])
                                 for i in others)),
                )] += 1
        if tot == 0:
            continue
        grand_tot += tot
        grand_ok += ok
        frac = ok / tot
        floor = FLOORS.get(path.stem, DEFAULT_FLOOR)
        status = "" if frac >= floor else f"  BELOW FLOOR {floor:.0%}"
        if frac < floor:
            failures.append(f"{path.stem}: {frac:.1%} < floor {floor:.0%}")
        print(f"{path.stem:22s} {tot:6d} {ok:7d} {frac:6.1%}{status}")
        for key, count in unmapped.most_common(1):
            print(f"{'':22s}   unmapped eg: stereo={key[0]} answers={key[1]} x{count}")

    print(f"\nTOTAL usable {grand_ok}/{grand_tot} ({grand_ok/grand_tot:.1%}) "
          f"across {len(list(DATA.glob('*.jsonl')))} categories")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASSED: every category meets its coverage floor")


if __name__ == "__main__":
    main()
