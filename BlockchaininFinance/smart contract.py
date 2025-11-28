# Minimal smart contract example (Python-side simulation)
# - Demonstrates deploying a contract, calling methods, and contract state
# - Simple ERC20-like token contract and an escrow contract example
#
# Run: python3 smart_contract_example.py

import time

class SimpleTokenContract:
    def __init__(self, name, symbol, owner):
        self.name = name
        self.symbol = symbol
        self.owner = owner
        self.balances = {}
        self.total_supply = 0

    def mint(self, to, amount, caller):
        if caller != self.owner:
            raise PermissionError("only owner can mint")
        self.balances[to] = self.balances.get(to, 0) + amount
        self.total_supply += amount
        print(f"Minted {amount} {self.symbol} to {to}")

    def transfer(self, frm, to, amount):
        if self.balances.get(frm, 0) < amount:
            raise ValueError("insufficient balance")
        self.balances[frm] -= amount
        self.balances[to] = self.balances.get(to, 0) + amount
        print(f"Transferred {amount} {self.symbol} from {frm} to {to}")

class EscrowContract:
    def __init__(self, token_contract):
        self.token = token_contract
        self.escrows = {}  # escrow_id -> dict(state)

    def create_escrow(self, escrow_id, payer, payee, amount):
        # payer must have approved/transfered token to this contract in a real chain
        self.escrows[escrow_id] = {
            "payer": payer,
            "payee": payee,
            "amount": amount,
            "released": False
        }
        print(f"Escrow {escrow_id} created: {payer} -> {payee} amount {amount}")

    def release(self, escrow_id, caller):
        e = self.escrows.get(escrow_id)
        if not e:
            raise KeyError("escrow not found")
        if e["released"]:
            print("already released")
            return
        if caller != e["payer"]:
            raise PermissionError("only payer can release in this demo")
        # Simulate token transfer from contract (we assume contract balance sufficient)
        self.token.transfer("contract", e["payee"], e["amount"])
        e["released"] = True
        print(f"Escrow {escrow_id} released to {e['payee']}")

if __name__ == "__main__":
    # Demo: deploy token, mint to Alice, create escrow for Bob
    token = SimpleTokenContract("DemoToken", "DMT", owner="Alice")
    token.mint("Alice", 1000, caller="Alice")
    # Contract receives some tokens in practice; simulate contract wallet
    token.balances["contract"] = 0

    escrow = EscrowContract(token)
    # Alice funds escrow by transferring to contract
    token.transfer("Alice", "contract", 200)
    escrow.create_escrow("escrow1", payer="Alice", payee="Bob", amount=200)
    # Later Alice releases funds to Bob
    escrow.release("escrow1", caller="Alice")
    print("Token balances:", token.balances)
