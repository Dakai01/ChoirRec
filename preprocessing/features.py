"""Parses the user log and exposure samples into time-windowed features."""
# update date：2026-06-07

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

REFERENCE_DATE = datetime(2026, 5, 30)

SHORT_TERM_DAYS = 7
MEDIUM_TERM_DAYS = 30
LONG_TERM_DAYS = 365

TOP_L1 = 20
TOP_LEAF = 20
TOP_SEARCH = 20

GENDERS = ["male", "female", "other", "unknown"]

AGE_UNKNOWN = -1


@dataclass
class Behavior:
    ts: datetime
    item_name: str
    l1_category: str
    leaf_category: str
    # Price tier in [0, 1] (1 = high tier, 0.5 = median). Only populated for purchase events.
    price: float = 0.5


@dataclass
class Search:
    ts: datetime
    query: str


@dataclass
class UserRecord:
    user_id: int
    age: int
    gender: str
    city: str
    clicks: List[Behavior] = field(default_factory=list)
    purchases: List[Behavior] = field(default_factory=list)
    searches: List[Search] = field(default_factory=list)


@dataclass
class L1Summary:
    l1_category: str
    count: int
    leaves: List[Tuple[str, int, float]]


@dataclass
class UserFeatures:
    """Aggregated, window-bucketed features used downstream."""
    user_id: int
    age: int
    gender: str
    city: str
    short_purchases: List[L1Summary]
    medium_purchases: List[L1Summary]
    long_purchases: List[L1Summary]
    recent_searches: List[Search]
    is_low_activity: int
    purchase_count: int
    click_count: int
    active_days: int


def _parse_dt(text: str) -> datetime:
    return datetime.strptime(text.strip(), "%Y-%m-%d %H:%M")


def _parse_behaviors(field_text: str) -> List[Behavior]:
    behaviors: List[Behavior] = []
    if not field_text.strip():
        return behaviors
    for entry in field_text.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) < 4:
            continue
        ts, name, l1, leaf = parts[0], parts[1], parts[2], parts[3]
        price = float(parts[4]) if len(parts) >= 5 and parts[4] else 0.5
        behaviors.append(Behavior(_parse_dt(ts), name, l1, leaf, price))
    return behaviors


def _parse_searches(field_text: str) -> List[Search]:
    searches: List[Search] = []
    if not field_text.strip():
        return searches
    for entry in field_text.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) < 2:
            continue
        searches.append(Search(_parse_dt(parts[0]), parts[1]))
    return searches


def load_user_records(path: str) -> List[UserRecord]:
    records: List[UserRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cells = line.split("\t")
            age_raw = cells[col["age"]].strip().lower()
            age = AGE_UNKNOWN if age_raw in ("", "unknown", "na", "null") else int(age_raw)
            records.append(
                UserRecord(
                    user_id=int(cells[col["user_id"]]),
                    age=age,
                    gender=cells[col["gender"]].strip(),
                    city=cells[col["city"]].strip(),
                    clicks=_parse_behaviors(cells[col["click_seq"]]),
                    purchases=_parse_behaviors(cells[col["purchase_seq"]]),
                    searches=_parse_searches(cells[col["search_seq"]]),
                )
            )
    return records


def _within_days(b_ts: datetime, days: int) -> bool:
    delta = REFERENCE_DATE - b_ts
    return 0 <= delta.days <= days


def _aggregate_top(behaviors: List[Behavior]) -> List[L1Summary]:
    """Top-20 L1 categories, each with its top-20 leaf categories."""
    l1_counter: Counter = Counter()
    leaf_by_l1: Dict[str, Counter] = {}
    price_by_leaf: Dict[str, List[float]] = {}
    for b in behaviors:
        l1_counter[b.l1_category] += 1
        leaf_by_l1.setdefault(b.l1_category, Counter())[b.leaf_category] += 1
        price_by_leaf.setdefault(b.leaf_category, []).append(b.price)

    summaries: List[L1Summary] = []
    for l1, cnt in l1_counter.most_common(TOP_L1):
        leaves: List[Tuple[str, int, float]] = []
        for leaf, lc in leaf_by_l1[l1].most_common(TOP_LEAF):
            prices = price_by_leaf.get(leaf, [0.5])
            leaves.append((leaf, lc, round(sum(prices) / len(prices), 2)))
        summaries.append(L1Summary(l1_category=l1, count=cnt, leaves=leaves))
    return summaries


def build_user_features(records: List[UserRecord]) -> List[UserFeatures]:
    features: List[UserFeatures] = []
    for r in records:
        short_clicks_raw = [b for b in r.clicks if _within_days(b.ts, SHORT_TERM_DAYS)]
        short_buy_raw = [b for b in r.purchases if _within_days(b.ts, SHORT_TERM_DAYS)]
        medium_buy_raw = [b for b in r.purchases if _within_days(b.ts, MEDIUM_TERM_DAYS)]
        long_buy_raw = [b for b in r.purchases if _within_days(b.ts, LONG_TERM_DAYS)]

        recent_searches = sorted(r.searches, key=lambda s: s.ts, reverse=True)[:TOP_SEARCH]

        # Low-activity flag. In production this should combine the user's
        # overall platform activity with the frequency of their various
        # behaviours; here it is simplified to a short-term behaviour count.
        recent_signal = len(short_clicks_raw) + len(short_buy_raw)
        is_low = 1 if recent_signal <= 2 else 0

        purchase_count = len(r.purchases)
        click_count = len(r.clicks)
        active_days = len({b.ts.date() for b in (list(r.clicks) + list(r.purchases))})

        features.append(
            UserFeatures(
                user_id=r.user_id,
                age=r.age,
                gender=r.gender if r.gender in GENDERS else "unknown",
                city=r.city,
                short_purchases=_aggregate_top(short_buy_raw),
                medium_purchases=_aggregate_top(medium_buy_raw),
                long_purchases=_aggregate_top(long_buy_raw),
                recent_searches=recent_searches,
                is_low_activity=is_low,
                purchase_count=purchase_count,
                click_count=click_count,
                active_days=active_days,
            )
        )
    return features


@dataclass
class Vocabularies:
    l1: Dict[str, int]
    leaf: Dict[str, int]
    city: Dict[str, int]
    item: Dict[str, int]
    user: Dict[str, int]

    def size_l1(self) -> int:
        return len(self.l1) + 1

    def size_leaf(self) -> int:
        return len(self.leaf) + 1

    def size_city(self) -> int:
        return len(self.city) + 1

    def size_item(self) -> int:
        return len(self.item) + 1

    def size_user(self) -> int:
        return len(self.user) + 1


def build_vocabularies(records: List[UserRecord]) -> Vocabularies:
    l1: "OrderedDict[str, int]" = OrderedDict()
    leaf: "OrderedDict[str, int]" = OrderedDict()
    city: "OrderedDict[str, int]" = OrderedDict()
    item: "OrderedDict[str, int]" = OrderedDict()
    user: "OrderedDict[str, int]" = OrderedDict()

    def add(d, key):
        if key not in d:
            d[key] = len(d) + 1  # ids start at 1; 0 reserved for PAD

    for r in records:
        add(city, r.city)
        add(user, str(r.user_id))
        for b in list(r.clicks) + list(r.purchases):
            add(l1, b.l1_category)
            add(leaf, b.leaf_category)
            add(item, b.item_name)
    return Vocabularies(l1=dict(l1), leaf=dict(leaf), city=dict(city),
                        item=dict(item), user=dict(user))


@dataclass
class Sample:
    user_idx: int
    item_name: str
    l1_category: str
    leaf_category: str
    label: int


def load_exposure_samples(path: str, records: List[UserRecord]) -> List[Sample]:
    user_index = {r.user_id: idx for idx, r in enumerate(records)}
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cells = line.split("\t")
            uid = int(cells[col["user_id"]])
            if uid not in user_index:
                continue
            samples.append(
                Sample(
                    user_idx=user_index[uid],
                    item_name=cells[col["item_name"]],
                    l1_category=cells[col["l1_category"]],
                    leaf_category=cells[col["leaf_category"]],
                    label=int(cells[col["label"]]),
                )
            )
    return samples
