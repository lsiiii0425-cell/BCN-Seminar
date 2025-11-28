# Tokenized bond lifecycle demo
# - Issue a bond, track coupons, allow transfers, and redeem at maturity
# - Very small simulation to show how financial instruments can be tokenized
#
# Run: python3 bonds_example.py

import time
from datetime import datetime, timedelta

class BondToken:
    def __init__(self, issuer, face_value, coupon_rate, maturity_days):
        self.issuer = issuer
        self.face_value = face_value
        self.coupon_rate = coupon_rate  # annual rate, e.g., 0.05
        self.maturity = datetime.utcnow() + timedelta(days=maturity_days)
        self.holders = {}  # holder -> amount (number of bond tokens)
        self.issued = False

    def issue(self, to, quantity):
        # Mint bond tokens to buyer
        self.holders[to] = self.holders.get(to, 0) + quantity
        self.issued = True
        print(f"Issued {quantity} bond(s) (face {self.face_value}) to {to}")

    def pay_coupon(self, now=None):
        now = now or datetime.utcnow()
        if now >= self.maturity:
            print("Bond matured — pay principal on redeem")
            return
        # For demo assume coupon paid yearly and proportional to quantity
        for holder, qty in self.holders.items():
            payment = qty * self.face_value * self.coupon_rate
            print(f"Coupon paid to {holder}: {payment:.2f} (qty {qty})")

    def transfer(self, frm, to, qty):
        if self.holders.get(frm, 0) < qty:
            raise ValueError("insufficient bond tokens")
        self.holders[frm] -= qty
        self.holders[to] = self.holders.get(to, 0) + qty
        print(f"Transferred {qty} bond token(s) from {frm} to {to}")

    def redeem(self, holder, now=None):
        now = now or datetime.utcnow()
        if now < self.maturity:
            raise ValueError("cannot redeem before maturity")
        qty = self.holders.get(holder, 0)
        if qty == 0:
            print("No bonds to redeem")
            return
        # Pay principal
        principal = qty * self.face_value
        print(f"Redeemed {qty} bond(s) from {holder}. Principal paid: {principal:.2f}")
        self.holders[holder] = 0

if __name__ == "__main__":
    # Demo: Issuer creates and issues bond tokens
    issuer = "AcmeCorp"
    bond = BondToken(issuer, face_value=1000, coupon_rate=0.05, maturity_days=30)
    bond.issue("InvestorA", 10)  # InvestorA gets 10 bonds -> face 10 * 1000
    bond.issue("InvestorB", 5)

    print("Holders:", bond.holders)
    # Simulate coupon payment now
    bond.pay_coupon()

    # Transfer a bond token
    bond.transfer("InvestorA", "InvestorC", qty=2)
    print("Holders after transfer:", bond.holders)

    # Fast-forward to maturity and redeem
    future = bond.maturity + timedelta(seconds=1)
    bond.redeem("InvestorA", now=future)
    bond.redeem("InvestorB", now=future)
    print("Holders after redemption:", bond.holders)
