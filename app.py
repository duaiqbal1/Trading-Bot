from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit
import pandas as pd
import trading_logic
import MetaTrader5 as mt5
import os
import random
import json
import logging
from threading import Lock
from datetime import datetime
import yfinance as yf
from werkzeug.security import generate_password_hash, check_password_hash

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
socketio = SocketIO(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simulated user database with hashed passwords
users = {'admin': {'password': generate_password_hash('password123')}}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Simulated account data (load from JSON file if it exists)
account_data_file = 'account_data.json'
if os.path.exists(account_data_file):
    try:
        with open(account_data_file, 'r') as f:
            account_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading account_data.json: {str(e)}")
        account_data = {
            "balance": 5000.0,
            "profit": 0.0,
            "open_trades": 0,
            "latest_signal": "SELL",
            "daily_high": 1.0600,
            "daily_low": 1.0345,
            "current_price": 1.0537,
            "bot_status": "STOPPED",
            "parameters": {
                "lot_size": 0.1,
                "stop_loss": 50,
                "take_profit": 100
            }
        }
else:
    account_data = {
        "balance": 5000.0,
        "profit": 0.0,
        "open_trades": 0,
        "latest_signal": "SELL",
        "daily_high": 1.0600,
        "daily_low": 1.0345,
        "current_price": 1.0537,
        "bot_status": "STOPPED",
        "parameters": {
            "lot_size": 0.1,
            "stop_loss": 50,
            "take_profit": 100
        }
    }
    try:
        with open(account_data_file, 'w') as f:
            json.dump(account_data, f)
    except Exception as e:
        logger.error(f"Error creating account_data.json: {str(e)}")

# Simulated trade history (will be updated dynamically)
trades = [
    {"time": "2025-03-23 14:22:57", "type": "SELL", "volume": 0.0, "profit": 0.0}
]

# Simulated profit and balance data (will be updated dynamically)
profit_data = [{"time": "2025-03-23 14:22:57", "profit": 0.0}]
balance_data = [{"time": "2025-03-23 14:22:57", "balance": 5000.0}]

# Simulated user database (replace with a real database in production)
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@app.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html', 
                          account=account_data, 
                          trades=trades, 
                          profit_data_json=json.dumps(profit_data), 
                          balance_data_json=json.dumps(balance_data))

@app.route('/update_parameters', methods=['POST'])
@login_required
def update_parameters():
    try:
        lot_size = float(request.form['lot_size'])
        stop_loss = int(request.form['stop_loss'])
        take_profit = int(request.form['take_profit'])

        # Validate parameters
        if lot_size <= 0:
            flash('Lot size must be greater than 0', 'error')
            return redirect(url_for('dashboard'))
        if stop_loss <= 0:
            flash('Stop loss must be greater than 0', 'error')
            return redirect(url_for('dashboard'))
        if take_profit <= 0:
            flash('Take profit must be greater than 0', 'error')
            return redirect(url_for('dashboard'))

        account_data["parameters"]["lot_size"] = lot_size
        account_data["parameters"]["stop_loss"] = stop_loss
        account_data["parameters"]["take_profit"] = take_profit
        with open(account_data_file, 'w') as f:
            json.dump(account_data, f)
        logger.debug(f"Parameters updated: {account_data['parameters']}")
        flash('Parameters updated successfully', 'success')
        return redirect(url_for('dashboard'))
    except ValueError as e:
        logger.error(f"Invalid input for parameters: {str(e)}")
        flash('Please enter valid numbers for parameters', 'error')
        return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"Error updating parameters: {str(e)}")
        flash('Error updating parameters', 'error')
        return redirect(url_for('dashboard'))

@app.route('/start_bot', methods=['POST'])
@login_required
def start_bot():
    try:
        logger.debug("Starting bot...")
        account_data["bot_status"] = "RUNNING"
        signal = trading_logic.main()  # Generate a new signal
        signal_file_path = os.getenv('TRADE_SIGNAL_PATH', os.path.join(os.getcwd(), 'trade_signal.txt'))
        try:
            with open(signal_file_path, 'r') as f:
                account_data["latest_signal"] = f.read().strip()
        except FileNotFoundError:
            logger.warning("trade_signal.txt not found, creating it with default signal 'HOLD'")
            with open(signal_file_path, 'w') as f:
                f.write("HOLD")
            account_data["latest_signal"] = "HOLD"
        with open(account_data_file, 'w') as f:
            json.dump(account_data, f)
        logger.debug(f"Bot status updated to: {account_data['bot_status']}, Latest signal: {account_data['latest_signal']}")
        return jsonify({"status": "Bot started successfully", "bot_status": account_data["bot_status"]})
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        return jsonify({"status": "Error starting bot", "error": str(e)}), 500

@app.route('/stop_bot', methods=['POST'])
@login_required
def stop_bot():
    try:
        logger.debug("Stopping bot...")
        account_data["bot_status"] = "STOPPED"
        if mt5.initialize():
            positions = mt5.positions_get(symbol="EURUSD")
            for position in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": position.ticket,
                    "symbol": "EURUSD",
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick("EURUSD").bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick("EURUSD").ask,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
            mt5.shutdown()
        with open(account_data_file, 'w') as f:
            json.dump(account_data, f)
        logger.debug(f"Bot status updated to: {account_data['bot_status']}")
        return jsonify({"status": "Bot stopped successfully", "bot_status": account_data["bot_status"]})
    except Exception as e:
        logger.error(f"Error stopping bot: {str(e)}")
        return jsonify({"status": "Error stopping bot", "error": str(e)}), 500

@app.route('/get_trades')
@login_required
def get_trades():
    try:
        logger.debug("Fetching trades...")
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Error fetching trades: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_account_data')
@login_required
def get_account_data():
    try:
        logger.debug("Fetching account data...")
        return jsonify({
            "balance": account_data["balance"],
            "profit": account_data["profit"],
            "open_trades": account_data["open_trades"]
        })
    except Exception as e:
        logger.error(f"Error fetching account data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_chart_data')
@login_required
def get_chart_data():
    try:
        logger.debug("Fetching chart data...")
        return jsonify({
            "profit_data": profit_data,
            "balance_data": balance_data
        })
    except Exception as e:
        logger.error(f"Error fetching chart data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/export_trades')
@login_required
def export_trades():
    try:
        df = pd.DataFrame(trades)
        csv_path = "trades_export.csv"
        df.to_csv(csv_path, index=False)
        logger.debug("Trades exported as CSV")
        return send_file(csv_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error exporting trades: {str(e)}")
        flash('Error exporting trades')
        return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username]['password'], password):
            user = User(username)
            login_user(user)
            logger.debug(f"User {username} logged in")
            return redirect(url_for('dashboard'))
        else:
            logger.warning(f"Failed login attempt for username: {username}")
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logger.debug(f"User {current_user.id} logged out")
    logout_user()
    return redirect(url_for('login'))

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not current_password or not new_password or not confirm_password:
            flash('All fields are required.', 'error')
            return redirect(url_for('change_password'))
        
        username = current_user.id
        if not check_password_hash(users[username]['password'], current_password):
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('change_password'))
        
        if new_password != confirm_password:
            flash('New password and confirmation do not match.', 'error')
            return redirect(url_for('change_password'))
        
        users[username]['password'] = generate_password_hash(new_password)
        logger.debug(f"Password updated for user {username}")
        flash('Password changed successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('change_password.html')

# Background task for real-time price updates and trade execution
thread = None
thread_lock = Lock()

@socketio.on('connect')
def handle_connect():
    global thread
    logger.debug("Client connected to SocketIO")
    with thread_lock:
        if thread is None:
            logger.debug("Starting background task for price updates")
            thread = socketio.start_background_task(price_update_task)

def price_update_task():
    while True:
        try:
            eurusd = yf.Ticker("EURUSD=X")
            data = eurusd.history(period="1d")
            current_price = data['Close'].iloc[-1]
            daily_high = data['High'].max()
            daily_low = data['Low'].min()
            account_data["current_price"] = round(current_price, 4)
            account_data["daily_high"] = round(daily_high, 4)
            account_data["daily_low"] = round(daily_low, 4)

            if mt5.initialize():
                account_info = mt5.account_info()
                if account_info:
                    account_data["balance"] = account_info.balance
                    account_data["profit"] = account_info.profit
                    account_data["open_trades"] = len(mt5.positions_get())
                else:
                    logger.error("Failed to fetch account info from MT5")
                mt5.shutdown()
            else:
                logger.error("Failed to connect to MT5 for account data")

            socketio.emit('price_update', {
                'current_price': account_data["current_price"],
                'daily_high': account_data["daily_high"],
                'daily_low': account_data["daily_low"]
            })
            logger.debug(f"Price updated: {account_data['current_price']}, High: {account_data['daily_high']}, Low: {account_data['daily_low']}")

            if account_data["bot_status"] == "RUNNING":
                execute_trade()

            with open(account_data_file, 'w') as f:
                json.dump(account_data, f)

            socketio.sleep(5)  # Update every 5 seconds

        except Exception as e:
            logger.error(f"Error in price update task: {str(e)}")
            socketio.sleep(5)

def execute_trade():
    """
    Execute a real trade on MT5 based on the latest signal.
    Updates trades, profit_data, balance_data, and account data from MT5.
    """
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            if not mt5.initialize():
                logger.error("Failed to connect to MT5 for trade execution")
                return

            if not mt5.symbol_select("EURUSD", True):
                logger.error("EURUSD symbol not available in MT5")
                mt5.shutdown()
                return

            trade_type = account_data["latest_signal"]
            if trade_type not in ["BUY", "SELL"]:
                logger.warning(f"Invalid trade signal: {trade_type}, skipping trade")
                mt5.shutdown()
                return

            volume = account_data["parameters"]["lot_size"]
            stop_loss = account_data["parameters"]["stop_loss"]
            take_profit = account_data["parameters"]["take_profit"]

            symbol_info = mt5.symbol_info_tick("EURUSD")
            if symbol_info is None:
                logger.error("Failed to fetch EURUSD price for trade execution")
                mt5.shutdown()
                return

            price = symbol_info.ask if trade_type == "BUY" else symbol_info.bid

            symbol_info = mt5.symbol_info("EURUSD")
            if symbol_info is None:
                logger.error("Failed to fetch EURUSD symbol info")
                mt5.shutdown()
                return

            point = symbol_info.point
            sl = price - stop_loss * point if trade_type == "BUY" else price + stop_loss * point
            tp = price + take_profit * point if trade_type == "BUY" else price - take_profit * point

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": "EURUSD",
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if trade_type == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Trade failed: {result.comment}")
                if "Requote" in result.comment and attempt < max_retries - 1:
                    logger.info(f"Retrying trade execution (attempt {attempt + 2}/{max_retries})...")
                    mt5.shutdown()
                    socketio.sleep(retry_delay)
                    continue
                mt5.shutdown()
                return

            trade_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            position_ticket = result.deal  # Use result.deal as the position ticket
            positions = mt5.positions_get(symbol="EURUSD")
            profit = 0.0
            matched_position_ticket = None
            for pos in positions:
                if pos.ticket == position_ticket:
                    profit = pos.profit
                    matched_position_ticket = pos.ticket
                    break

            new_trade = {
                "time": trade_time,
                "type": trade_type,
                "volume": volume,
                "profit": round(profit, 2),
                "position_ticket": matched_position_ticket
            }
            trades.append(new_trade)

            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to fetch account info from MT5")
                mt5.shutdown()
                return

            account_data["balance"] = account_info.balance
            account_data["profit"] = account_info.profit
            account_data["open_trades"] = len(mt5.positions_get())

            profit_data.append({"time": trade_time, "profit": round(account_data["profit"], 2)})
            balance_data.append({"time": trade_time, "balance": round(account_data["balance"], 2)})

            logger.debug(f"Executed trade: {new_trade}")
            break

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying trade execution (attempt {attempt + 2}/{max_retries})...")
                socketio.sleep(retry_delay)
                continue
            break
        finally:
            mt5.shutdown()

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)