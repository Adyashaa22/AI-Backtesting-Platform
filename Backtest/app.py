from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
from flask_cors import CORS
import random
from datetime import datetime, timedelta
import threading
import time
import csv
import os
import shutil
import zipfile
from io import BytesIO

"""
Multi-Page Flask Backtesting Platform - FIXED FOR LARGE DATASETS
Page 1: Home (Paper Trading / Backtesting)
Page 2: Backtest Configuration
Page 3: Results with Metrics
Page 4: Price Chart Page
Page 5: Equity Chart Page

Run: python app.py
Then open: http://localhost:5000
"""
import pyotp

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
import json
from io import StringIO
import os
from datetime import datetime, timedelta
import secrets
try:
    from SmartApi import SmartConnect
    SMARTAPI_AVAILABLE = True
except ImportError:
    SMARTAPI_AVAILABLE = False
    print("⚠️  SmartAPI not installed. Install with: pip install smartapi-python")

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # For session management

# -----------------------------
# CSV Writer Module
# -----------------------------
class CSVWriter:
    def __init__(self, base_dir="trading_data"):
        self.base_dir = base_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, self.session_id)
        
        # Create directory structure
        os.makedirs(self.session_dir, exist_ok=True)
        
        # File paths
        self.candles_file = os.path.join(self.session_dir, "candles.csv")
        self.trades_file = os.path.join(self.session_dir, "trades.csv")
        self.summary_file = os.path.join(self.session_dir, "session_summary.csv")
        
        # Initialize files with headers
        self._init_candles_file()
        self._init_trades_file()
        
        print(f"✓ CSV files initialized in: {self.session_dir}")
    
    def _init_candles_file(self):
        """Initialize candles CSV with headers"""
        with open(self.candles_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Open", "High", "Low", "Close", "Volume (Ticks)"
            ])
    
    def _init_trades_file(self):
        """Initialize trades CSV with headers"""
        with open(self.trades_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trade ID", "Action", "Timestamp", "Price", "Quantity",
                "Entry Price", "Exit Price", "Entry Time", "Exit Time",
                "Holding Period (min)", "Gross P&L", "Gross P&L %", 
                "Fees", "Net P&L", "Net P&L %", "Status"
            ])
    
    def write_candle(self, bar, volume):
        """Write a single candle to CSV"""
        try:
            with open(self.candles_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    bar["timestamp"],
                    bar["open"],
                    bar["high"],
                    bar["low"],
                    bar["close"],
                    volume
                ])
                f.flush()  # Force write to disk
        except Exception as e:
            print(f"Error writing candle: {e}")
            import traceback
            traceback.print_exc()
    
    def write_trade(self, action_data):
        """Write trade action to CSV"""
        try:
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if action_data["action"] == "buy":
                    writer.writerow([
                        action_data.get("trade_id", ""),
                        "BUY",
                        action_data["timestamp"],
                        action_data["price"],
                        action_data.get("quantity", ""),
                        action_data.get("entry_price", ""),
                        "",
                        action_data["timestamp"],
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "OPEN"
                    ])
                
                elif action_data["action"] == "sell":
                    writer.writerow([
                        action_data.get("trade_id", ""),
                        "SELL",
                        action_data["timestamp"],
                        action_data["price"],
                        action_data.get("quantity", ""),
                        action_data.get("entry_price", ""),
                        action_data.get("exit_price", ""),
                        action_data.get("entry_time", ""),
                        action_data["timestamp"],
                        action_data.get("holding_period_min", ""),
                        action_data.get("pnl_gross", ""),
                        action_data.get("pnl_percent_gross", ""),
                        action_data.get("fees", ""),
                        action_data.get("pnl", ""),
                        action_data.get("pnl_percent", ""),
                        action_data.get("status", "")
                    ])
        except Exception as e:
            print(f"Error writing trade: {e}")
    
    def write_session_summary(self, settings, metrics, trade_stats, engine):
        """Write final session summary"""
        try:
            with open(self.summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow(["=== SESSION INFORMATION ==="])
                writer.writerow(["Session ID", self.session_id])
                writer.writerow(["Stock", settings["stock"]])
                writer.writerow(["Timeframe (min)", settings["timeframe"]])
                writer.writerow(["Brokerage (%)", settings["brokerage"]])
                writer.writerow(["Order Quantity", settings["quantity"]])
                writer.writerow([])
                
                writer.writerow(["=== ACCOUNT METRICS ==="])
                writer.writerow(["Initial Balance", settings["initial_balance"]])
                writer.writerow(["Final Balance", engine.balance])
                writer.writerow(["Portfolio Value", metrics["portfolio_value"]])
                writer.writerow(["Total P&L", metrics["pnl"]])
                writer.writerow(["Total P&L %", metrics["pnl_percent"]])
                writer.writerow(["Current Position", metrics["current_position"]])
                writer.writerow([])
                
                writer.writerow(["=== TRADING STATISTICS ==="])
                writer.writerow(["Total Trades", trade_stats["total_closed_trades"]])
                writer.writerow(["Winning Trades", trade_stats["winning_trades"]])
                writer.writerow(["Losing Trades", trade_stats["losing_trades"]])
                writer.writerow(["Win Rate (%)", trade_stats["win_rate"]])
                writer.writerow(["Avg Win", trade_stats["avg_win"]])
                writer.writerow(["Avg Loss", trade_stats["avg_loss"]])
                writer.writerow(["Largest Win", trade_stats["largest_win"]])
                writer.writerow(["Largest Loss", trade_stats["largest_loss"]])
                writer.writerow(["Profit Factor", trade_stats["profit_factor"]])
                writer.writerow(["Expectancy", trade_stats["expectancy"]])
                writer.writerow(["Avg Hold Time (min)", trade_stats["avg_holding_period"]])
                writer.writerow([])
                
                writer.writerow(["=== RISK METRICS ==="])
                writer.writerow(["Sharpe Ratio", metrics["sharpe_ratio"]])
                writer.writerow(["Max Drawdown (%)", metrics["max_drawdown"]])
                writer.writerow([])
                
                writer.writerow(["=== GENERATED ==="])
                writer.writerow(["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            
            print(f"✓ Session summary saved: {self.summary_file}")
        except Exception as e:
            print(f"Error writing session summary: {e}")
    
    def create_zip_archive(self):
        """Create a ZIP file containing all CSV files"""
        try:
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(self.candles_file, os.path.basename(self.candles_file))
                zipf.write(self.trades_file, os.path.basename(self.trades_file))
                zipf.write(self.summary_file, os.path.basename(self.summary_file))
            
            memory_file.seek(0)
            return memory_file
        except Exception as e:
            print(f"Error creating ZIP archive: {e}")
            return None


# -----------------------------
# LiveDataAggregator Module
# -----------------------------
class LiveDataAggregator:
    def __init__(self, interval_minutes=1):
        self.interval_minutes = interval_minutes
        self.ticks = []
        self.start_time = None
        self.tick_count = 0

    def add_tick(self, tick):
        ltp = float(tick['ltp'])
        ts = tick['timestamp']

        if self.start_time is None:
            self.start_time = ts
        self.ticks.append(ltp)
        self.tick_count += 1

        elapsed = (ts - self.start_time).total_seconds()
        if elapsed >= self.interval_minutes * 60:
            bar = self._aggregate_bar()
            volume = self.tick_count
            self.ticks = [ltp]
            self.start_time = ts
            self.tick_count = 1
            return bar, volume
        return None, 0

    def _aggregate_bar(self):
        return {
            "open": self.ticks[0],
            "high": max(self.ticks),
            "low": min(self.ticks),
            "close": self.ticks[-1],
            "timestamp": self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        }


# -----------------------------
# PaperTradingEngine Module
# -----------------------------
class PaperTradingEngine:
    def __init__(self, initial_balance=100000, brokerage_percent=0.0, slippage_percent=0.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.avg_price = 0
        self.trade_log = []
        self.equity_curve = []
        self.peak_value = initial_balance
        self.max_drawdown = 0
        self.brokerage_percent = float(brokerage_percent)
        self.slippage_percent = 0.0
        
        self.completed_trades = []
        self.open_trade = None
        self.trade_id_counter = 0
        self.last_error = None

    def execute_trade(self, action, price, quantity=1):
        if action == "buy":
            if self.open_trade is not None:
                return False
                
            effective_entry_price = price
            gross_cost = effective_entry_price * quantity
            buy_fee = gross_cost * (self.brokerage_percent / 100.0)
            total_cost = gross_cost + buy_fee
            if self.balance >= total_cost:
                self.balance -= total_cost
                old_position = self.position
                self.position += quantity
                self.avg_price = ((self.avg_price * old_position) + gross_cost) / self.position
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.trade_log.append({
                    "action": "buy", 
                    "price": effective_entry_price, 
                    "quantity": quantity, 
                    "timestamp": timestamp
                })
                
                self.trade_id_counter += 1
                self.open_trade = {
                    "trade_id": self.trade_id_counter,
                    "entry_price": effective_entry_price,
                    "entry_time": timestamp,
                    "quantity": quantity,
                    "type": "LONG",
                    "buy_fee": round(buy_fee, 2)
                }
                
                return True
            else:
                self.last_error = {
                    "type": "INSUFFICIENT_FUNDS",
                    "message": f"Insufficient balance for BUY: need ₹{total_cost:.2f}, have ₹{self.balance:.2f}",
                    "required": round(total_cost, 2),
                    "available": round(self.balance, 2)
                }
                return False
                
        elif action == "sell":
            if self.open_trade is None:
                self.last_error = {"type": "NO_OPEN_TRADE", "message": "No open position to sell"}
                return False
                
            if self.position >= quantity:
                effective_exit_price = price
                gross_proceeds = effective_exit_price * quantity
                sell_fee = gross_proceeds * (self.brokerage_percent / 100.0)
                net_proceeds = gross_proceeds - sell_fee
                self.balance += net_proceeds
                self.position -= quantity
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.trade_log.append({
                    "action": "sell", 
                    "price": effective_exit_price, 
                    "quantity": quantity, 
                    "timestamp": timestamp
                })
                
                entry_price = self.open_trade["entry_price"]
                pnl_gross = (effective_exit_price - entry_price) * quantity
                total_fees = (self.open_trade.get("buy_fee", 0.0)) + sell_fee
                pnl = pnl_gross - total_fees
                entry_notional = entry_price * quantity if entry_price != 0 else 0
                pnl_percent = (pnl / entry_notional * 100) if entry_notional else 0
                pnl_percent_gross = (pnl_gross / entry_notional * 100) if entry_notional else 0
                
                entry_dt = datetime.strptime(self.open_trade["entry_time"], "%Y-%m-%d %H:%M:%S")
                exit_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                holding_period = (exit_dt - entry_dt).total_seconds() / 60
                
                completed_trade = {
                    "trade_id": self.open_trade["trade_id"],
                    "type": self.open_trade["type"],
                    "entry_price": entry_price,
                    "exit_price": effective_exit_price,
                    "quantity": quantity,
                    "entry_time": self.open_trade["entry_time"],
                    "exit_time": timestamp,
                    "holding_period_min": round(holding_period, 2),
                    "pnl": round(pnl, 2),
                    "pnl_percent": round(pnl_percent, 2),
                    "pnl_gross": round(pnl_gross, 2),
                    "pnl_percent_gross": round(pnl_percent_gross, 2),
                    "fees": round(total_fees, 2),
                    "status": "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
                }
                
                self.completed_trades.append(completed_trade)
                self.open_trade = None
                
                return True
            else:
                self.last_error = {"type": "INSUFFICIENT_QUANTITY", "message": "Not enough quantity to sell"}
                return False
        return False

    def get_portfolio_value(self, current_price):
        value = self.balance + self.position * current_price
        self.equity_curve.append(value)
        
        if value > self.peak_value:
            self.peak_value = value
        
        drawdown = (self.peak_value - value) / self.peak_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
        return value

    def get_trade_statistics(self):
        if not self.completed_trades:
            return {
                "total_closed_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "avg_holding_period": 0,
                "profit_factor": 0,
                "expectancy": 0
            }
        
        wins = [t for t in self.completed_trades if t["pnl"] > 0]
        losses = [t for t in self.completed_trades if t["pnl"] < 0]
        
        total_wins = sum(t["pnl"] for t in wins)
        total_losses = abs(sum(t["pnl"] for t in losses))
        
        return {
            "total_closed_trades": len(self.completed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round((len(wins) / len(self.completed_trades)) * 100, 2),
            "avg_win": round(total_wins / len(wins), 2) if wins else 0,
            "avg_loss": round(total_losses / len(losses), 2) if losses else 0,
            "largest_win": round(max((t["pnl"] for t in wins), default=0), 2),
            "largest_loss": round(min((t["pnl"] for t in losses), default=0), 2),
            "avg_holding_period": round(sum(t["holding_period_min"] for t in self.completed_trades) / len(self.completed_trades), 2),
            "profit_factor": round(total_wins / total_losses, 2) if total_losses > 0 else 0,
            "expectancy": round(sum(t["pnl"] for t in self.completed_trades) / len(self.completed_trades), 2)
        }

    def get_metrics(self, current_price):
        portfolio_value = self.get_portfolio_value(current_price)
        pnl = portfolio_value - self.initial_balance
        pnl_percent = (pnl / self.initial_balance) * 100
        
        if len(self.equity_curve) > 1:
            returns = [(self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1] 
                      for i in range(1, len(self.equity_curve))]
            if returns:
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                sharpe_ratio = (avg_return / std_return * (252 ** 0.5)) if std_return != 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        return {
            "portfolio_value": round(portfolio_value, 2),
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl_percent, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(self.max_drawdown * 100, 2),
            "total_trades": len(self.trade_log),
            "current_position": self.position,
            "balance": round(self.balance, 2)
        }


# -----------------------------
# Random Agent Module
# -----------------------------
class RandomAgent_paper:
    def act(self, state):
        return random.choice(["buy", "sell", "hold"])


# -----------------------------
# Flask App
# -----------------------------

CORS(app)

STOCK_START_PRICES = {
    "RELIANCE": 2500.0,
    "TCS": 3800.0,
    "INFY": 1600.0,
    "HDFCBANK": 1500.0,
    "ICICIBANK": 1200.0,
    "SBIN": 750.0,
    "HINDUNILVR": 2500.0,
    "ITC": 470.0,
    "BHARTIARTL": 1200.0,
    "AXISBANK": 1100.0,
}

SETTINGS = {
    "stock": "RELIANCE",
    "initial_balance": 100000.0,
    "timeframe": 1,
    "brokerage": 0.05,
    "slippage": 0.0,
    "quantity": 10,
}

# Global state
aggregator = None
engine = None
agent = None
csv_writer = None
candles_data = []
current_ltp = STOCK_START_PRICES.get(SETTINGS["stock"], 190.0)
agent_actions = []
default_order_qty = SETTINGS["quantity"]
tick_interval = 12

# Trading state management
trading_active = False
market_thread = None
stop_trading_event = threading.Event()

def apply_settings():
    global aggregator, engine, candles_data, agent_actions, current_ltp, default_order_qty, csv_writer, agent
    candles_data = []
    agent_actions = []
    aggregator = LiveDataAggregator(interval_minutes=SETTINGS["timeframe"])
    engine = PaperTradingEngine(
        initial_balance=SETTINGS["initial_balance"],
        brokerage_percent=SETTINGS["brokerage"],
        slippage_percent=0.0,
    )
    agent = RandomAgent_paper()
    csv_writer = CSVWriter()
    default_order_qty = SETTINGS["quantity"]
    current_ltp = STOCK_START_PRICES.get(SETTINGS["stock"], 190.0)

def simulate_market():
    global current_ltp, candles_data, agent_actions, trading_active
    current_time = datetime.now()
    candle_counter = 0
    summary_update_interval = 10
    
    while not stop_trading_event.is_set():
        try:
            ##current_time = datetime.now()
            current_ltp = round(current_ltp + random.uniform(-0.5, 0.5), 2)
            
            tick = {
                "ltp": current_ltp,
                "timestamp": current_time
            }
            
            bar, volume = aggregator.add_tick(tick)
            if bar:
                candles_data.append(bar)
                candle_counter += 1
                
                csv_writer.write_candle(bar, volume)
                
                if len(candles_data) > 10000:
                    candles_data.pop(0)
                    for action in agent_actions:
                        if 'candle_index' in action:
                            action['candle_index'] -= 1
                
                print(f"Candle #{len(candles_data)}: O:{bar['open']:.2f} H:{bar['high']:.2f} L:{bar['low']:.2f} C:{bar['close']:.2f} V:{volume}")
                
                action = agent.act(bar)
                
                if action != "hold":
                    success = engine.execute_trade(action, bar['close'], quantity=default_order_qty)
                    if success:
                        print(f"✓ Trade: {action.upper()} @ ₹{bar['close']:.2f}")
                        
                        action_data = {
                            "timestamp": bar['timestamp'],
                            "action": action,
                            "price": bar['close'],
                            "candle_index": len(candles_data) - 1
                        }
                        
                        if action == "buy" and engine.open_trade:
                            action_data["trade_id"] = engine.open_trade["trade_id"]
                            action_data["entry_price"] = engine.open_trade["entry_price"]
                            action_data["quantity"] = engine.open_trade["quantity"]
                            action_data["entry_time"] = engine.open_trade["entry_time"]
                        elif action == "sell" and len(engine.completed_trades) > 0:
                            last_trade = engine.completed_trades[-1]
                            action_data.update({
                                "trade_id": last_trade["trade_id"],
                                "entry_price": last_trade["entry_price"],
                                "exit_price": last_trade["exit_price"],
                                "quantity": last_trade["quantity"],
                                "pnl": last_trade["pnl"],
                                "pnl_percent": last_trade["pnl_percent"],
                                "pnl_gross": last_trade["pnl_gross"],
                                "pnl_percent_gross": last_trade["pnl_percent_gross"],
                                "holding_period_min": last_trade["holding_period_min"],
                                "fees": last_trade["fees"],
                                "status": last_trade["status"],
                                "entry_time": last_trade["entry_time"]
                            })
                        
                        csv_writer.write_trade(action_data)
                        agent_actions.append(action_data)
                        
                        if len(agent_actions) > 10000:
                            agent_actions.pop(0)
                
                if candle_counter % summary_update_interval == 0:
                    try:
                        metrics = engine.get_metrics(current_ltp)
                        trade_stats = engine.get_trade_statistics()
                        csv_writer.write_session_summary(SETTINGS, metrics, trade_stats, engine)
                        print(f"📊 Summary updated (Candle #{len(candles_data)})")
                    except Exception as e:
                        print(f"Error updating summary: {e}")
            current_time = current_time + timedelta(seconds=20)
            time.sleep(1)
        except Exception as e:
            print(f"Error in simulate_market: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
    
    print("Market simulation stopped")
    trading_active = False


@app.route('/')
def index():
    # If trading is active, redirect to dashboard
    if trading_active:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/start', methods=['POST'])
def start():
    global trading_active, market_thread, stop_trading_event
    
    try:
        # If already trading, redirect to dashboard
        if trading_active:
            return redirect(url_for('dashboard'))
        
        stock = request.form.get('stock', SETTINGS["stock"]).strip()
        initial_balance = float(request.form.get('initial_balance', SETTINGS["initial_balance"]))
        timeframe = int(request.form.get('timeframe', SETTINGS["timeframe"]))
        brokerage = float(request.form.get('brokerage', SETTINGS["brokerage"]))
        quantity = int(request.form.get('quantity', SETTINGS["quantity"]))

        SETTINGS.update({
            "stock": stock,
            "initial_balance": initial_balance,
            "timeframe": timeframe,
            "brokerage": brokerage,
            "slippage": 0.0,
            "quantity": quantity,
        })

        apply_settings()
        
        # Start trading
        trading_active = True
        stop_trading_event.clear()
        market_thread = threading.Thread(target=simulate_market, daemon=True)
        market_thread.start()
        
        print("\n" + "="*60)
        print("📊 PAPER TRADING ENGINE STARTED")
        print("="*60)
        print(f"CSV Output Directory: trading_data/{csv_writer.session_id}/")
        print("="*60 + "\n")
        
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Error in /start: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/dashboard')
def dashboard():
    # If trading is not active, redirect to config
    if not trading_active:
        return redirect(url_for('index'))
    return render_template('dashboard.html')

@app.route('/api/stop_trading', methods=['POST'])
def stop_trading():
    global trading_active, stop_trading_event
    
    try:
        if not trading_active:
            return jsonify({"error": "Trading is not active"}), 400
        
        # Generate final summary
        metrics = engine.get_metrics(current_ltp)
        trade_stats = engine.get_trade_statistics()
        csv_writer.write_session_summary(SETTINGS, metrics, trade_stats, engine)
        
        # Create ZIP file
        zip_file = csv_writer.create_zip_archive()
        
        if zip_file is None:
            return jsonify({"error": "Failed to create ZIP archive"}), 500
        
        # Signal the market thread to stop
        stop_trading_event.set()
        
        # Return the ZIP file
        filename = f"trading_session_{csv_writer.session_id}.zip"
        
        return send_file(
            zip_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Error stopping trading: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/data')
def get_data():
    try:
        if not trading_active:
            return jsonify({"error": "Trading not active"}), 400
            
        start = request.args.get('start', type=int)
        end = request.args.get('end', type=int)
        
        total_candles = len(candles_data)
        
        if start is None or end is None:
            start = max(0, total_candles - 500)
            end = total_candles
        
        start = max(0, min(start, total_candles))
        end = max(start, min(end, total_candles))
        
        windowed_candles = candles_data
        
        current_candle = None
        if aggregator.ticks and aggregator.start_time:
            current_candle = {
                "open": aggregator.ticks[0],
                "high": max(aggregator.ticks),
                "low": min(aggregator.ticks),
                "close": aggregator.ticks[-1],
                "timestamp": aggregator.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "is_forming": True
            }
        
        windowed_actions = agent_actions
        latest_candles = candles_data[-10:] if total_candles > 10 else candles_data
        
        metrics = engine.get_metrics(current_ltp)
        trade_stats = engine.get_trade_statistics()
        
        return jsonify({
            "candles": windowed_candles,
            "actions": windowed_actions,
            "start_index": start,
            "end_index": end,
            "total_candles": total_candles,
            "latest_candles": latest_candles,
            "current_candle": current_candle,
            "current_ltp": current_ltp,
            "metrics": metrics,
            "trade_stats": trade_stats,
            "completed_trades": engine.completed_trades[-100:],
            "open_trade": engine.open_trade,
            "settings": SETTINGS,
            "last_error": engine.last_error,
            "csv_session_id": csv_writer.session_id
        })
    except Exception as e:
        print(f"Error in /api/data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/trading_status')
def trading_status():
    """Check if trading is currently active"""
    return jsonify({"trading_active": trading_active})

# Configuration
POINTS_PER_PAGE = 1000
MAX_FILE_SIZE_MB = 100
TRADES_PER_PAGE = 100  # For trade log pagination

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """Backtesting engine for AI trading agents"""
    
    def __init__(self, initial_capital=100000, quantity_per_trade=10, 
                 transaction_cost_pct=0.1, slippage_pct=0.05):
        self.initial_capital = initial_capital
        self.quantity_per_trade = quantity_per_trade
        self.transaction_cost_pct = transaction_cost_pct / 100
        self.slippage_pct = slippage_pct / 100
        
        self.cash = initial_capital
        self.position = 0
        self.position_entry_price = 0
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
    
    def reset(self):
        self.cash = self.initial_capital
        self.position = 0
        self.position_entry_price = 0
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
    
    def get_execution_price(self, price, action):
        if action == 1:
            return price * (1 + self.slippage_pct)
        elif action == -1:
            return price * (1 - self.slippage_pct)
        return price
    
    def calculate_transaction_cost(self, trade_value):
        return trade_value * self.transaction_cost_pct
    
    def execute_trade_backtest(self, action, price, timestamp):
        trade_details = None
        execution_price = self.get_execution_price(price, action)
        
        if action == 1:  # BUY
            if self.position == 0:
                trade_value = execution_price * self.quantity_per_trade
                transaction_cost = self.calculate_transaction_cost(trade_value)
                total_cost = trade_value + transaction_cost
                
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.position = self.quantity_per_trade
                    self.position_entry_price = execution_price
                    
                    trade_details = {
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'quantity': self.quantity_per_trade,
                        'price': execution_price,
                        'trade_value': trade_value,
                        'transaction_cost': transaction_cost,
                        'total_cost': total_cost,
                        'cash_after': self.cash
                    }
        
        elif action == -1:  # SELL
            if self.position > 0:
                trade_value = execution_price * self.position
                transaction_cost = self.calculate_transaction_cost(trade_value)
                net_proceeds = trade_value - transaction_cost
                
                pnl = (execution_price - self.position_entry_price) * self.position
                pnl_pct = ((execution_price - self.position_entry_price) / self.position_entry_price) * 100
                
                self.cash += net_proceeds
                
                trade_details = {
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'quantity': self.position,
                    'price': execution_price,
                    'entry_price': self.position_entry_price,
                    'trade_value': trade_value,
                    'transaction_cost': transaction_cost,
                    'net_proceeds': net_proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'cash_after': self.cash
                }
                
                self.position = 0
                self.position_entry_price = 0
        
        return trade_details
    
    def get_portfolio_value(self, current_price):
        position_value = self.position * current_price
        return self.cash + position_value
    
    def run_backtest(self, data, agent):
        """Run backtest on FULL dataset"""
        self.reset()
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"⚙️  Processing {len(data)} bars...")
        
        for i in range(len(data) - 1):
            current_bar = data.iloc[i]
            next_bar = data.iloc[i + 1]
            
            action = agent.get_action(current_bar)
            execution_price = next_bar['close']
            trade_details = self.execute_trade_backtest(action, execution_price, next_bar['timestamp'])
            
            if trade_details:
                self.trades.append(trade_details)
            
            portfolio_value = self.get_portfolio_value(execution_price)
            self.equity_curve.append({
                'timestamp': next_bar['timestamp'],
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'position': self.position,
                'price': execution_price
            })
            
            if len(self.equity_curve) > 1:
                prev_value = self.equity_curve[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
            
            if (i + 1) % 50000 == 0:
                print(f"   Processed {i+1:,} / {len(data):,} bars ({((i+1)/len(data)*100):.1f}%)")
        
        if self.position > 0:
            final_price = data.iloc[-1]['close']
            final_trade = self.execute_trade_backtest(-1, final_price, data.iloc[-1]['timestamp'])
            if final_trade:
                self.trades.append(final_trade)
        
        metrics = self.calculate_metrics()
        
        print(f"✅ Backtest complete!")
        
        return {
            'metrics': metrics,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
            'data': data
        }
    
    def calculate_metrics(self):
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        final_value = equity_df['portfolio_value'].iloc[-1]
        
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        if len(self.daily_returns) > 0:
            returns_array = np.array(self.daily_returns)
            excess_returns = returns_array - (0.06 / 252)
            sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / returns_array.std()) if returns_array.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        equity_values = equity_df['portfolio_value'].values
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            if not sell_trades.empty:
                total_trades = len(sell_trades)
                winning_trades = len(sell_trades[sell_trades['pnl'] > 0])
                losing_trades = len(sell_trades[sell_trades['pnl'] < 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                avg_win = sell_trades[sell_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = sell_trades[sell_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
                avg_pnl = sell_trades['pnl'].mean()
                total_pnl = sell_trades['pnl'].sum()
                
                profit_factor = abs(sell_trades[sell_trades['pnl'] > 0]['pnl'].sum() / 
                                   sell_trades[sell_trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
            else:
                total_trades = winning_trades = losing_trades = 0
                win_rate = avg_win = avg_loss = avg_pnl = profit_factor = total_pnl = 0
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = avg_pnl = profit_factor = total_pnl = 0
        
        return {
            'initialCapital': self.initial_capital,
            'finalValue': final_value,
            'totalReturn': total_return,
            'sharpeRatio': sharpe_ratio,
            'maxDrawdown': max_drawdown,
            'totalTrades': total_trades,
            'winningTrades': winning_trades,
            'losingTrades': losing_trades,
            'winRate': win_rate,
            'avgWin': avg_win,
            'avgLoss': avg_loss,
            'avgPnL': avg_pnl,
            'totalPnL': total_pnl,
            'profitFactor': profit_factor if profit_factor != float('inf') else 999
        }


class RandomAgent:
    """Random agent for testing"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def get_action(self, bar):
        return np.random.choice([1, 0, -1], p=[0.15, 0.70, 0.15])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def paginate_data(df, trades_df, page=0, points_per_page=POINTS_PER_PAGE):
    """Get paginated data for charts"""
    start_idx = page * points_per_page
    end_idx = start_idx + points_per_page
    
    paginated_df = df.iloc[start_idx:end_idx].copy()
    
    if not trades_df.empty:
        page_timestamps = set(paginated_df['timestamp'])
        page_trades = trades_df[trades_df['timestamp'].isin(page_timestamps)]
    else:
        page_trades = pd.DataFrame()
    
    return paginated_df, page_trades


def map_trades_to_chart(trades_df, chart_data):
    """Map trades to chart data"""
    buy_trades = []
    sell_trades = []
    
    if trades_df.empty:
        return buy_trades, sell_trades
    
    chart_timestamps_set = set(chart_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
    
    for _, trade in trades_df.iterrows():
        trade_timestamp = trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        if trade_timestamp in chart_timestamps_set:
            trade_point = {
                'timestamp': trade_timestamp,
                'action': trade['action'],
                'price': float(trade['price']),
                'quantity': int(trade.get('quantity', 0)),
                'pnl': float(trade.get('pnl', 0)) if pd.notna(trade.get('pnl')) else None,
                'pnl_pct': float(trade.get('pnl_pct', 0)) if pd.notna(trade.get('pnl_pct')) else None
            }
            
            if trade['action'] == 'BUY':
                buy_trades.append(trade_point)
            elif trade['action'] == 'SELL':
                sell_trades.append(trade_point)
    
    return buy_trades, sell_trades


def fetch_smartapi_data(symbol, interval, days_back=30):
    """
    Fetch historical data from SmartAPI (Angel One)
    
    Args:
        symbol: Trading symbol (e.g., 'RELIANCE-EQ')
        interval: Time interval ('1m', '5m', '15m', '30m', '1h', '1d')
        days_back: Number of days of historical data to fetch
    
    Returns:
        pandas DataFrame with columns: timestamp, close
    """
    if not SMARTAPI_AVAILABLE:
        raise Exception("SmartAPI library not installed. Please install with: pip install smartapi-python")
    
    try:
        # Initialize SmartAPI connection
        # Note: You'll need to set API_KEY and CLIENT_ID in environment variables or config
        api_key = "7Z0QEGp7"
        client_id = "AACA031085"
        password = "1674"
        totp = pyotp.TOTP("B2SH2CTGFWMXONFGKK6NTTWLL4").now() 
        
        if not api_key:
            # For demo purposes, we'll use a mock data approach
            # In production, you need to provide actual credentials
            print("⚠️  SmartAPI credentials not configured. Using mock data generator.")
            return generate_mock_data(symbol, interval, days_back)
        
        obj = SmartConnect(api_key)
        
        # Login (if credentials provided)
        if client_id and password and totp:
            data = obj.generateSession(client_id, password, totp)
            if data['status']:
                print("✅ SmartAPI login successful")
            else:
                raise Exception(f"SmartAPI login failed: {data.get('message', 'Unknown error')}")
        
        # Map interval to SmartAPI format
        interval_map = {
            '1m': 'ONE_MINUTE',
            '3m': 'THREE_MINUTE',
            '5m': 'FIVE_MINUTE',
            '10m': 'TEN_MINUTE',
            '15m': 'FIFTEEN_MINUTE',
            '30m': 'THIRTY_MINUTE',
            '1h': 'ONE_HOUR',
            '1d': 'ONE_DAY'
        }
        
        smartapi_interval = interval_map.get(interval, 'FIVE_MINUTE')
        
        # SmartAPI official limits for maximum days per request
        # Source: SmartAPI documentation
        max_days_map = {
            '1m': 30,    # ONE_MINUTE: 30 days
            '3m': 60,    # THREE_MINUTE: 60 days
            '5m': 100,   # FIVE_MINUTE: 100 days
            '10m': 100,  # TEN_MINUTE: 100 days
            '15m': 200,  # FIFTEEN_MINUTE: 200 days
            '30m': 200,  # THIRTY_MINUTE: 200 days
            '1h': 400,   # ONE_HOUR: 400 days
            '1d': 2000   # ONE_DAY: 2000 days
        }
        
        # Adjust days_back to respect SmartAPI limits
        max_days = max_days_map.get(interval, 100)  # Default to 100 if interval not found
        if days_back > max_days:
            print(f"⚠️  Limiting to {max_days} days for {interval} interval (SmartAPI limit)")
            days_back = max_days
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Extract symbol name (remove -EQ suffix if present)
        symbol_name = symbol.replace('-EQ', '') if '-EQ' in symbol else symbol
        
        # Symbol to token mapping for common NSE stocks
        # These are approximate token values - SmartAPI may require exact tokens
        symbol_token_map = {
            'RELIANCE': '2885',
            'TCS': '11536',
            'INFY': '408065',
            'HDFCBANK': '1333',
            'SBIN': '3045',
            'ICICIBANK': '4963',
            'ITC': '4243',
            'AXISBANK': '5900',
            'LT': '2933',
            'WIPRO': '9685'
        }
        
        # Try to get token from mapping first
        symbol_token = symbol_token_map.get(symbol_name.upper(), None)
        
        if symbol_token:
            print(f"✅ Using token {symbol_token} for {symbol_name} from mapping")
        else:
            # Try to search for symbol using SmartAPI
            print(f"🔍 Searching for symbol: {symbol_name}")
            search_result = None
            try:
                # Try different possible method signatures
                if hasattr(obj, 'searchScrip'):
                    # Try with different parameter names
                    try:
                        search_result = obj.searchScrip(exchange="NSE", symbolname=symbol_name)
                    except (TypeError, AttributeError):
                        try:
                            search_result = obj.searchScrip(exchange="NSE", symbol=symbol_name)
                        except (TypeError, AttributeError):
                            try:
                                # Some versions use searchstring
                                search_result = obj.searchScrip(exchange="NSE", searchstring=symbol_name)
                            except (TypeError, AttributeError):
                                pass
                
                if search_result and 'data' in search_result and len(search_result['data']) > 0:
                    # Get the first matching symbol
                    symbol_info = search_result['data'][0]
                    symbol_token = symbol_info.get('token', None)
                    if symbol_token:
                        print(f"✅ Found symbol token: {symbol_token} for {symbol_info.get('symbol', symbol_name)}")
            except Exception as e:
                print(f"⚠️  Symbol search failed: {e}")
            
            # If still no token, use symbol name (might work for some APIs)
            if not symbol_token:
                print(f"⚠️  Could not find symbol token for {symbol_name}, will try using symbol name")
                symbol_token = symbol_name
        
        # Format dates - SmartAPI expects specific format
        # For intraday, use time; for daily, just date
        if interval == '1d':
            fromdate_str = start_date.strftime("%Y-%m-%d 09:15")
            todate_str = end_date.strftime("%Y-%m-%d 15:30")
        else:
            fromdate_str = start_date.strftime("%Y-%m-%d %H:%M")
            todate_str = end_date.strftime("%Y-%m-%d %H:%M")
        
        # Fetch historical data
        historicParam = {
            "exchange": "NSE",
            "symboltoken": str(symbol_token),  # Ensure it's a string
            "interval": smartapi_interval,
            "fromdate": fromdate_str,
            "todate": todate_str
        }
        
        print(f"📊 Requesting data: {historicParam}")
        response = obj.getCandleData(historicParam)
        
        # Check response
        if not response:
            raise Exception("Empty response from SmartAPI")
        
        if not response.get('status', False):
            error_msg = response.get('message', 'Unknown error')
            error_code = response.get('errorcode', '')
            print(f"❌ SmartAPI error: {error_msg} (Code: {error_code})")
            raise Exception(f"SmartAPI error: {error_msg}")
        
        if 'data' not in response or not response['data']:
            raise Exception(f"No data returned from SmartAPI: {response}")
        
        # Convert to DataFrame
        candles = response['data']
        if len(candles) == 0:
            raise Exception("Empty data array from SmartAPI")
        
        # SmartAPI returns data in format: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp - SmartAPI returns in different formats
        try:
            # Try ISO format first
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S')
        except:
            try:
                # Try with milliseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
            except:
                # Try generic parsing
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Select only required columns
        df = df[['timestamp', 'close']].copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Fetched {len(df)} data points from SmartAPI")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching SmartAPI data: {str(e)}")
        # Fallback to mock data if SmartAPI fails
        print("⚠️  Falling back to mock data generator")
        return generate_mock_data(symbol, interval, days_back)


def generate_mock_data(symbol, interval, days_back=30):
    """
    Generate mock historical data for testing when SmartAPI is not available
    """
    print(f"📊 Generating mock data for {symbol} with {interval} interval")
    
    # Map interval to minutes
    interval_minutes = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '10m': 10,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '1d': 1440
    }
    
    minutes = interval_minutes.get(interval, 5)
    total_minutes = days_back * 24 * 60
    num_points = total_minutes // minutes
    
    # Generate timestamps
    end_time = datetime.now()
    timestamps = [end_time - timedelta(minutes=minutes * i) for i in range(num_points, 0, -1)]
    
    # Generate realistic price data (random walk)
    np.random.seed(42)
    base_price = 2000.0  # Base price
    prices = [base_price]
    
    for i in range(1, num_points):
        change = np.random.normal(0, 10)  # Random walk
        new_price = max(100, prices[-1] + change)  # Ensure price doesn't go below 100
        prices.append(new_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices
    })
    
    print(f"✅ Generated {len(df)} mock data points")
    return df


# Store backtest results in memory
backtest_cache = {}


# ============================================================================
# FLASK ROUTES
# ============================================================================



@app.route('/backtest')
def backtest():
    if trading_active:
        return redirect(url_for('dashboard'))
    return render_template('backtest.html')

@app.route('/fetchdata')
def fetch():
    if trading_active:
        return redirect(url_for('dashboard'))
    return render_template('fetchdata.html')

@app.route('/papertrading')
def papertrading():
    if trading_active:
        return redirect(url_for('dashboard'))
    """Paper Trading - Not implemented yet"""
    return render_template('config.html')


@app.route('/backtesting')
def backtesting():
    if trading_active:
        return redirect(url_for('dashboard'))
    """Page 2: Backtest configuration page"""
    return render_template('backtesting.html')


@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    if trading_active:
        return redirect(url_for('dashboard'))
    """Run backtest and redirect to results page"""
    try:
        start_time = datetime.now()
        
        price_file = request.files.get('priceData')
        if not price_file:
            return jsonify({'error': 'Price data file is required'}), 400
        
        price_file.seek(0, 2)
        file_size = price_file.tell()
        price_file.seek(0)
        
        max_size = MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_size:
            return jsonify({'error': f'File too large. Maximum {MAX_FILE_SIZE_MB}MB allowed'}), 400
        
        print(f"\n{'='*80}")
        print(f"📊 BACKTEST STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
        
        initial_capital = float(request.form.get('initialCapital', 100000))
        quantity = int(request.form.get('quantity', 10))
        transaction_cost = float(request.form.get('transactionCost', 0.1))
        slippage = float(request.form.get('slippage', 0.05))
        
        print("📥 Reading CSV file...")
        price_data = pd.read_csv(price_file)
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        print(f"✅ Loaded {len(price_data):,} data points")
        print(f"📅 Date range: {price_data['timestamp'].min()} to {price_data['timestamp'].max()}")
        
        required_cols = ['timestamp', 'close']
        if not all(col in price_data.columns for col in required_cols):
            return jsonify({'error': f'CSV must contain: {", ".join(required_cols)}'}), 400
        
        engine = BacktestEngine(
            initial_capital=initial_capital,
            quantity_per_trade=quantity,
            transaction_cost_pct=transaction_cost,
            slippage_pct=slippage
        )
        
        agent = RandomAgent(seed=42)
        
        print(f"\n🚀 Running backtest on {len(price_data):,} bars...")
        results = engine.run_backtest(price_data, agent)
        
        backtest_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️  Backtest completed in {backtest_time:.2f} seconds")
        print(f"💼 Total trades executed: {len(results['trades'])}")
        
        session_id = str(datetime.now().timestamp())
        backtest_cache[session_id] = {
            'results': results,
            'backtest_time': backtest_time
        }
        session['current_backtest'] = session_id
        
        total_points = len(results['data'])
        total_pages = (total_points + POINTS_PER_PAGE - 1) // POINTS_PER_PAGE
        
        print(f"📊 Total pages: {total_pages} ({POINTS_PER_PAGE} points per page)")
        
        # FIXED: Don't send full trade log in initial response
        response_data = {
            'sessionId': session_id,
            'metrics': results['metrics'],
            'tradeLog': [],  # Empty - will be loaded separately via /api/get_trades
            'dataInfo': {
                'totalDataPoints': len(results['data']),
                'totalTrades': len(results['trades']),
                'backtestTime': round(backtest_time, 2),
                'dateRange': {
                    'start': results['data']['timestamp'].min().strftime('%Y-%m-%d'),
                    'end': results['data']['timestamp'].max().strftime('%Y-%m-%d')
                }
            },
            'pagination': {
                'totalPages': total_pages,
                'pointsPerPage': POINTS_PER_PAGE,
                'totalPoints': total_points
            }
        }
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Response prepared in {total_time:.2f} seconds total")
        print(f"{'='*80}\n")
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch_and_run_backtest', methods=['POST'])
def fetch_and_run_backtest():
    if trading_active:
        return redirect(url_for('dashboard'))
    """Fetch data from SmartAPI and run backtest"""
    try:
        start_time = datetime.now()
        
        # Get parameters from request
        symbol = request.json.get('symbol')
        timeframe = request.json.get('timeframe')
        initial_capital = float(request.json.get('initialCapital', 100000))
        quantity = int(request.json.get('quantity', 10))
        transaction_cost = float(request.json.get('transactionCost', 0.1))
        slippage = float(request.json.get('slippage', 0.05))
        days_back = int(request.json.get('daysBack', 30))
        
        if not symbol or not timeframe:
            return jsonify({'error': 'Symbol and timeframe are required'}), 400
        
        print(f"\n{'='*80}")
        print(f"📊 SMARTAPI BACKTEST STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"📈 Symbol: {symbol}")
        print(f"⏱️  Timeframe: {timeframe}")
        print(f"📅 Days back: {days_back}")
        
        # Fetch data from SmartAPI
        print("📥 Fetching data from SmartAPI...")
        try:
            price_data = fetch_smartapi_data(symbol, timeframe, days_back)
        except Exception as fetch_error:
            print(f"⚠️  Error in fetch_smartapi_data: {fetch_error}")
            # Fallback to mock data
            print("📊 Using mock data generator as fallback")
            price_data = generate_mock_data(symbol, timeframe, days_back)
        
        if price_data.empty:
            print("⚠️  Received empty data, generating mock data")
            price_data = generate_mock_data(symbol, timeframe, days_back)
        
        print(f"✅ Loaded {len(price_data):,} data points")
        print(f"📅 Date range: {price_data['timestamp'].min()} to {price_data['timestamp'].max()}")
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in price_data.columns:
            return jsonify({'error': 'Data must contain timestamp column'}), 400
        
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        # Ensure close column exists
        if 'close' not in price_data.columns:
            return jsonify({'error': 'Data must contain close column'}), 400
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=initial_capital,
            quantity_per_trade=quantity,
            transaction_cost_pct=transaction_cost,
            slippage_pct=slippage
        )
        
        agent = RandomAgent(seed=42)
        
        print(f"\n🚀 Running backtest on {len(price_data):,} bars...")
        results = engine.run_backtest(price_data, agent)
        
        backtest_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️  Backtest completed in {backtest_time:.2f} seconds")
        print(f"💼 Total trades executed: {len(results['trades'])}")
        
        # Store results in cache
        session_id = str(datetime.now().timestamp())
        backtest_cache[session_id] = {
            'results': results,
            'backtest_time': backtest_time
        }
        session['current_backtest'] = session_id
        
        total_points = len(results['data'])
        total_pages = (total_points + POINTS_PER_PAGE - 1) // POINTS_PER_PAGE
        
        print(f"📊 Total pages: {total_pages} ({POINTS_PER_PAGE} points per page)")
        
        response_data = {
            'sessionId': session_id,
            'metrics': results['metrics'],
            'tradeLog': [],
            'dataInfo': {
                'totalDataPoints': len(results['data']),
                'totalTrades': len(results['trades']),
                'backtestTime': round(backtest_time, 2),
                'dateRange': {
                    'start': results['data']['timestamp'].min().strftime('%Y-%m-%d'),
                    'end': results['data']['timestamp'].max().strftime('%Y-%m-%d')
                }
            },
            'pagination': {
                'totalPages': total_pages,
                'pointsPerPage': POINTS_PER_PAGE,
                'totalPoints': total_points
            }
        }
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Response prepared in {total_time:.2f} seconds total")
        print(f"{'='*80}\n")
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results():
    if trading_active:
        return redirect(url_for('dashboard'))
    """Page 3: Results page with metrics and buttons"""
    session_id = session.get('current_backtest')
    if not session_id or session_id not in backtest_cache:
        return redirect(url_for('backtesting'))
    
    return render_template('result.html')


@app.route('/price_chart')
def price_chart():
    if trading_active:
        return redirect(url_for('dashboard'))
    """Page 4: Price chart page"""
    session_id = session.get('current_backtest')
    if not session_id or session_id not in backtest_cache:
        return redirect(url_for('backtesting'))
    
    return render_template('price_chart.html')


@app.route('/equity_chart')
def equity_chart():
    if trading_active:
        return redirect(url_for('dashboard'))
    """Page 5: Equity chart page"""
    session_id = session.get('current_backtest')
    if not session_id or session_id not in backtest_cache:
        return redirect(url_for('backtesting'))
    
    return render_template('equity_chart.html')


@app.route('/api/get_results', methods=['GET'])
def get_results():
    if trading_active:
        return redirect(url_for('dashboard'))
    """API: Get current backtest results summary (WITHOUT full trade log for performance)"""
    try:
        session_id = session.get('current_backtest')
        if not session_id or session_id not in backtest_cache:
            return jsonify({'error': 'No backtest data found'}), 404
        
        cached_data = backtest_cache[session_id]
        results = cached_data['results']
        backtest_time = cached_data['backtest_time']
        
        total_points = len(results['data'])
        total_pages = (total_points + POINTS_PER_PAGE - 1) // POINTS_PER_PAGE
        
        # FIXED: Only send trade count, not full trade log (use /api/get_trades for that)
        total_trades = len(results['trades'])
        
        response_data = {
            'sessionId': session_id,
            'metrics': results['metrics'],
            'tradeLog': [],  # Empty - use /api/get_trades/<session_id>/<page> to load trades
            'dataInfo': {
                'totalDataPoints': len(results['data']),
                'totalTrades': total_trades,
                'backtestTime': round(backtest_time, 2),
                'dateRange': {
                    'start': results['data']['timestamp'].min().strftime('%Y-%m-%d'),
                    'end': results['data']['timestamp'].max().strftime('%Y-%m-%d')
                }
            },
            'pagination': {
                'totalPages': total_pages,
                'pointsPerPage': POINTS_PER_PAGE,
                'totalPoints': total_points
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_trades/<session_id>/<int:page>', methods=['GET'])
def get_trades_page(session_id, page):
    if trading_active:
        return redirect(url_for('dashboard'))
    """API: Get paginated trade log (NEW ENDPOINT for large datasets)"""
    try:
        if session_id not in backtest_cache:
            return jsonify({'error': 'Session expired'}), 404
        
        cached_data = backtest_cache[session_id]
        results = cached_data['results']
        
        if results['trades'].empty:
            return jsonify({
                'trades': [],
                'pagination': {
                    'currentPage': 0,
                    'totalPages': 0,
                    'tradesPerPage': TRADES_PER_PAGE,
                    'totalTrades': 0
                }
            })
        
        total_trades = len(results['trades'])
        total_pages = (total_trades + TRADES_PER_PAGE - 1) // TRADES_PER_PAGE
        
        if page < 0 or page >= total_pages:
            return jsonify({'error': 'Invalid page number'}), 400
        
        start_idx = page * TRADES_PER_PAGE
        end_idx = min(start_idx + TRADES_PER_PAGE, total_trades)
        
        page_trades = results['trades'].iloc[start_idx:end_idx]
        
        trade_log = []
        for _, trade in page_trades.iterrows():
            trade_log.append({
                'timestamp': trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'action': trade['action'],
                'price': float(trade['price']),
                'quantity': int(trade.get('quantity', 0)),
                'pnl': float(trade.get('pnl', 0)) if pd.notna(trade.get('pnl')) else None,
                'pnl_pct': float(trade.get('pnl_pct', 0)) if pd.notna(trade.get('pnl_pct')) else None
            })
        
        return jsonify({
            'trades': trade_log,
            'pagination': {
                'currentPage': page,
                'totalPages': total_pages,
                'tradesPerPage': TRADES_PER_PAGE,
                'totalTrades': total_trades
            }
        })
    
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_page/<session_id>/<int:page>', methods=['GET'])
def get_page(session_id, page):
    if trading_active:
        return redirect(url_for('dashboard'))
    """API: Get specific page of chart data"""
    try:
        if session_id not in backtest_cache:
            return jsonify({'error': 'Session expired'}), 404
        
        cached_data = backtest_cache[session_id]
        results = cached_data['results']
        
        total_points = len(results['data'])
        total_pages = (total_points + POINTS_PER_PAGE - 1) // POINTS_PER_PAGE
        
        if page < 0 or page >= total_pages:
            return jsonify({'error': 'Invalid page number'}), 400
        
        page_price_data, page_trades = paginate_data(results['data'], results['trades'], page=page)
        page_equity_data, _ = paginate_data(results['equity_curve'], pd.DataFrame(), page=page)
        
        buy_trades, sell_trades = map_trades_to_chart(page_trades, page_price_data)
        
        response_data = {
            'priceData': {
                'timestamps': page_price_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'close': page_price_data['close'].tolist()
            },
            'equityCurve': {
                'timestamps': page_equity_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'values': page_equity_data['portfolio_value'].tolist()
            },
            'trades': {
                'buy': buy_trades,
                'sell': sell_trades
            },
            'pagination': {
                'currentPage': page,
                'totalPages': total_pages,
                'pointsPerPage': POINTS_PER_PAGE,
                'totalPoints': total_points
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_metrics/<session_id>')
def download_metrics(session_id):
    if trading_active:
        return redirect(url_for('dashboard'))
    """API: Download metrics CSV"""
    if session_id not in backtest_cache:
        return jsonify({'error': 'Session expired'}), 404
    
    cached_data = backtest_cache[session_id]
    results = cached_data['results']
    m = results['metrics']
    
    # Create CSV content
    csv_lines = [
        'Metric,Value',
        f'Initial Capital,₹{m["initialCapital"]}',
        f'Final Value,₹{m["finalValue"]:.2f}',
        f'Total Return,{m["totalReturn"]:.2f}%',
        f'Sharpe Ratio,{m["sharpeRatio"]:.2f}',
        f'Max Drawdown,{m["maxDrawdown"]:.2f}%',
        f'Total Trades,{m["totalTrades"]}',
        f'Winning Trades,{m["winningTrades"]}',
        f'Losing Trades,{m["losingTrades"]}',
        f'Win Rate,{m["winRate"]:.2f}%',
        f'Total P&L,₹{m["totalPnL"]:.2f}',
        f'Avg Win,₹{m["avgWin"]:.2f}',
        f'Avg Loss,₹{m["avgLoss"]:.2f}',
        f'Profit Factor,{m["profitFactor"] if m["profitFactor"] < 999 else "∞"}'
    ]
    
    return '\n'.join(csv_lines), 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=backtest_metrics.csv'
    }


@app.route('/api/download_tradelog/<session_id>')
def download_tradelog(session_id):
    if trading_active:
        return redirect(url_for('dashboard'))
    """API: Download trade log CSV"""
    if session_id not in backtest_cache:
        return jsonify({'error': 'Session expired'}), 404
    
    cached_data = backtest_cache[session_id]
    results = cached_data['results']
    
    if results['trades'].empty:
        return 'No trades to export', 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=trade_log.csv'
        }
    
    csv_lines = ['Timestamp,Action,Quantity,Price,P&L,P&L %']
    
    for _, trade in results['trades'].iterrows():
        pnl = f"{trade.get('pnl', 0):.2f}" if pd.notna(trade.get('pnl')) else ''
        pnl_pct = f"{trade.get('pnl_pct', 0):.2f}" if pd.notna(trade.get('pnl_pct')) else ''
        
        csv_lines.append(
            f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')},"
            f"{trade['action']},"
            f"{trade.get('quantity', 0)},"
            f"{trade['price']:.2f},"
            f"{pnl},"
            f"{pnl_pct}"
        )
    
    return '\n'.join(csv_lines), 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=trade_log.csv'
    }


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 80)
    print("🚀 Multi-Page AI Agent Backtesting Platform (OPTIMIZED FOR LARGE DATASETS)")
    print("=" * 80)
    print("\n📍 Open your browser: http://localhost:5000")
    print("\n📄 Pages:")
    print("   1. Home: Choose Paper Trading or Backtesting")
    print("   2. Backtest Config: Upload CSV & Configure")
    print("   3. Results: View Metrics & Access Charts")
    print("   4. Price Chart: Interactive price chart with signals")
    print("   5. Equity Chart: Portfolio value over time")
    print("\n💡 CSV Format: timestamp, close (minimum required)")
    print("📦 Max file size: 100 MB")
    print("🔧 Trade log pagination: 100 trades per page")
    print("=" * 80)
    print()
    

    app.run(debug=True, port=5000, use_reloader=False)