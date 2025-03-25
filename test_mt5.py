import MetaTrader5 as mt5

# Initialize MT5
if not mt5.initialize():
    print("Failed to initialize MT5")
    quit()

# Log in
if not mt5.login(login=5034556303, password="A_N7QzBx", server="MetaQuotes-Demo"):
    print(f"Login failed: {mt5.last_error()}")
    quit()

# Test account info
account_info = mt5.account_info()
if account_info is None:
    print(f"Failed to get account info: {mt5.last_error()}")
else:
    print(f"Account Balance: {account_info.balance}")

# Test positions total
trades = mt5.positions_total()
if trades is None:
    print(f"Failed to get positions: {mt5.last_error()}")
else:
    print(f"Open Trades: {trades}")