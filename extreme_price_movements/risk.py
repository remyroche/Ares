import numpy as np

class TrailingStop:
    def __init__(
        self,
        entry_px: float,
        side: str,
        atr_val: float,
        k_sl: float = 2.0,
        k_trail_start: float = 1.0,
        k_trail_dist: float = 1.0,
        score_conf: float = 0.0 # New param
    ):
        self.entry_px = entry_px
        self.side = side
        self.atr = atr_val

        # Adjust K params by score_conf?
        # "adjust the TP & SL values by confidence (backtest linear model's score weight too)"
        # Assuming k_sl becomes k_sl * (1 + score_conf * score)
        # But score is passed here? No, score_conf is the WEIGHT.
        # We need the actual score too? Or assume k_sl passed is ALREADY adjusted?
        # Engine calls this. Engine has score.
        # It's cleaner if Engine computes adjusted k values and passes them.
        # But if we want to serialize `score_conf` logic here...
        # Let's keep TrailingStop simple: It takes k_values.
        # Scaling logic belongs in Engine/Training.

        self.k_sl = k_sl
        self.k_trail_start = k_trail_start
        self.k_trail_dist = k_trail_dist

        self.initial_sl_dist = k_sl * atr_val * entry_px

        if side == "long":
            self.sl_px = entry_px - self.initial_sl_dist
            self.highest_high = entry_px
        else:
            self.sl_px = entry_px + self.initial_sl_dist
            self.lowest_low = entry_px

        self.trailing_active = False

    def update(self, current_high: float, current_low: float, current_close: float):
        if self.side == "long":
            stop_hit = current_low <= self.sl_px
            trail_would_trigger = False

            if current_high > self.highest_high:
                profit_dist = current_high - self.entry_px
                req_start_dist = self.k_trail_start * self.atr * self.entry_px
                is_active = self.trailing_active or (profit_dist >= req_start_dist)

                if is_active:
                    trail_dist_px = self.k_trail_dist * self.atr * self.entry_px
                    new_sl = current_high - trail_dist_px
                    if new_sl > self.sl_px:
                        trail_would_trigger = True

            if stop_hit and trail_would_trigger:
                return True, self.entry_px, "ambiguous_neutral"

            if stop_hit:
                return True, self.sl_px, "sl_hit"

            if current_high > self.highest_high:
                self.highest_high = current_high

            profit_dist = self.highest_high - self.entry_px
            req_start_dist = self.k_trail_start * self.atr * self.entry_px

            if profit_dist >= req_start_dist:
                self.trailing_active = True

            if self.trailing_active:
                trail_dist_px = self.k_trail_dist * self.atr * self.entry_px
                new_sl = self.highest_high - trail_dist_px
                if new_sl > self.sl_px:
                    self.sl_px = new_sl

        else: # short
            stop_hit = current_high >= self.sl_px
            trail_would_trigger = False

            if current_low < self.lowest_low:
                profit_dist = self.entry_px - current_low
                req_start_dist = self.k_trail_start * self.atr * self.entry_px
                is_active = self.trailing_active or (profit_dist >= req_start_dist)

                if is_active:
                    trail_dist_px = self.k_trail_dist * self.atr * self.entry_px
                    new_sl = current_low + trail_dist_px
                    if new_sl < self.sl_px:
                        trail_would_trigger = True

            if stop_hit and trail_would_trigger:
                return True, self.entry_px, "ambiguous_neutral"

            if stop_hit:
                return True, self.sl_px, "sl_hit"

            if current_low < self.lowest_low:
                self.lowest_low = current_low

            profit_dist = self.entry_px - self.lowest_low
            req_start_dist = self.k_trail_start * self.atr * self.entry_px

            if profit_dist >= req_start_dist:
                self.trailing_active = True

            if self.trailing_active:
                trail_dist_px = self.k_trail_dist * self.atr * self.entry_px
                new_sl = self.lowest_low + trail_dist_px
                if new_sl < self.sl_px:
                    self.sl_px = new_sl

        return False, None, None

    def get_sl_px(self):
        return self.sl_px

    def to_dict(self):
        return {
            "entry_px": self.entry_px,
            "side": self.side,
            "atr": self.atr,
            "k_sl": self.k_sl,
            "k_trail_start": self.k_trail_start,
            "k_trail_dist": self.k_trail_dist,
            "sl_px": self.sl_px,
            "highest_high": getattr(self, "highest_high", None),
            "lowest_low": getattr(self, "lowest_low", None),
            "trailing_active": self.trailing_active
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(
            entry_px=d["entry_px"],
            side=d["side"],
            atr_val=d["atr"],
            k_sl=d["k_sl"],
            k_trail_start=d["k_trail_start"],
            k_trail_dist=d["k_trail_dist"]
        )
        obj.sl_px = d["sl_px"]
        if "highest_high" in d and d["highest_high"] is not None:
            obj.highest_high = d["highest_high"]
        if "lowest_low" in d and d["lowest_low"] is not None:
            obj.lowest_low = d["lowest_low"]
        obj.trailing_active = d["trailing_active"]
        return obj
