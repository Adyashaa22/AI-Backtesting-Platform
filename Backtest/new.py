from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
import random
from datetime import datetime, timedelta
import threading
import time
import csv
import os

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
            with open(self.candles_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    bar["timestamp"],
                    bar["open"],
                    bar["high"],
                    bar["low"],
                    bar["close"],
                    volume
                ])
        except Exception as e:
            print(f"Error writing candle: {e}")
    
    def write_trade(self, action_data):
        """Write trade action to CSV"""
        try:
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if action_data["action"] == "buy":
                    # Buy action
                    writer.writerow([
                        action_data.get("trade_id", ""),
                        "BUY",
                        action_data["timestamp"],
                        action_data["price"],
                        action_data.get("quantity", ""),
                        action_data.get("entry_price", ""),
                        "",  # Exit price (empty for buy)
                        action_data["timestamp"],
                        "",  # Exit time (empty for buy)
                        "",  # Holding period
                        "",  # Gross P&L
                        "",  # Gross P&L %
                        "",  # Fees
                        "",  # Net P&L
                        "",  # Net P&L %
                        "OPEN"
                    ])
                
                elif action_data["action"] == "sell":
                    # Sell action - full trade details
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
                
                # Session Info
                writer.writerow(["=== SESSION INFORMATION ==="])
                writer.writerow(["Session ID", self.session_id])
                writer.writerow(["Stock", settings["stock"]])
                writer.writerow(["Timeframe (min)", settings["timeframe"]])
                writer.writerow(["Brokerage (%)", settings["brokerage"]])
                writer.writerow(["Order Quantity", settings["quantity"]])
                writer.writerow([])
                
                # Account Metrics
                writer.writerow(["=== ACCOUNT METRICS ==="])
                writer.writerow(["Initial Balance", settings["initial_balance"]])
                writer.writerow(["Final Balance", engine.balance])
                writer.writerow(["Portfolio Value", metrics["portfolio_value"]])
                writer.writerow(["Total P&L", metrics["pnl"]])
                writer.writerow(["Total P&L %", metrics["pnl_percent"]])
                writer.writerow(["Current Position", metrics["current_position"]])
                writer.writerow([])
                
                # Trading Statistics
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
                
                # Risk Metrics
                writer.writerow(["=== RISK METRICS ==="])
                writer.writerow(["Sharpe Ratio", metrics["sharpe_ratio"]])
                writer.writerow(["Max Drawdown (%)", metrics["max_drawdown"]])
                writer.writerow([])
                
                # Timestamp
                writer.writerow(["=== GENERATED ==="])
                writer.writerow(["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            
            print(f"✓ Session summary saved: {self.summary_file}")
        except Exception as e:
            print(f"Error writing session summary: {e}")


# -----------------------------
# LiveDataAggregator Module (Enhanced)
# -----------------------------
class LiveDataAggregator:
    def __init__(self, interval_minutes=1):
        self.interval_minutes = interval_minutes
        self.ticks = []
        self.start_time = None
        self.tick_count = 0  # Track volume

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
            self.ticks = [ltp]  # Include current tick in new candle
            self.start_time = ts
            self.tick_count = 1  # Reset with current tick
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
# PaperTradingEngine Module (Same as before)
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
class RandomAgent:
    def act(self, state):
        return random.choice(["buy", "sell", "hold"])


# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
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
aggregator = LiveDataAggregator(interval_minutes=SETTINGS["timeframe"])
engine = PaperTradingEngine(
    initial_balance=SETTINGS["initial_balance"],
    brokerage_percent=SETTINGS["brokerage"],
    slippage_percent=0.0,
)
agent = RandomAgent()
csv_writer = CSVWriter()

candles_data = []
current_ltp = STOCK_START_PRICES.get(SETTINGS["stock"], 190.0)
agent_actions = []
default_order_qty = SETTINGS["quantity"]
tick_interval = 12

def apply_settings():
    global aggregator, engine, candles_data, agent_actions, current_ltp, default_order_qty, csv_writer
    candles_data = []
    agent_actions = []
    aggregator = LiveDataAggregator(interval_minutes=SETTINGS["timeframe"])
    engine = PaperTradingEngine(
        initial_balance=SETTINGS["initial_balance"],
        brokerage_percent=SETTINGS["brokerage"],
        slippage_percent=0.0,
    )
    csv_writer = CSVWriter()  # New session
    default_order_qty = SETTINGS["quantity"]
    current_ltp = STOCK_START_PRICES.get(SETTINGS["stock"], 190.0)

def simulate_market():
    global current_ltp, candles_data, agent_actions
    current_time = datetime.now()
    candle_counter = 0
    summary_update_interval = 10  # Update summary every 10 candles
    
    while True:
        try:
            current_time = datetime.now()
            current_ltp = round(current_ltp + random.uniform(-0.5, 0.5), 2)
            
            tick = {
                "ltp": current_ltp,
                "timestamp": current_time
            }
            
            bar, volume = aggregator.add_tick(tick)
            if bar:
                candles_data.append(bar)
                candle_counter += 1
                
                # Save candle to CSV
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
                        
                        # Save trade to CSV
                        csv_writer.write_trade(action_data)
                        
                        agent_actions.append(action_data)
                        
                        if len(agent_actions) > 10000:
                            agent_actions.pop(0)
                
                # Update summary CSV periodically
                if candle_counter % summary_update_interval == 0:
                    try:
                        metrics = engine.get_metrics(current_ltp)
                        trade_stats = engine.get_trade_statistics()
                        csv_writer.write_session_summary(SETTINGS, metrics, trade_stats, engine)
                        print(f"📊 Summary updated (Candle #{len(candles_data)})")
                    except Exception as e:
                        print(f"Error updating summary: {e}")
            
            time.sleep(1)
        except Exception as e:
            print(f"Error in simulate_market: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


@app.route('/')
def index():
    return render_template('config.html')

@app.route('/start', methods=['POST'])
def start():
    try:
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
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Error in /start: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    try:
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

@app.route('/api/export_summary', methods=['POST'])
def export_summary():
    """Generate session summary CSV on demand"""
    try:
        metrics = engine.get_metrics(current_ltp)
        trade_stats = engine.get_trade_statistics()
        csv_writer.write_session_summary(SETTINGS, metrics, trade_stats, engine)
        return jsonify({
            "success": True,
            "message": f"Summary exported to {csv_writer.summary_file}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    market_thread = threading.Thread(target=simulate_market, daemon=True)
    market_thread.start()
    
    print("\n" + "="*60)
    print("📊 PAPER TRADING ENGINE STARTED")
    print("="*60)
    print(f"CSV Output Directory: trading_data/{csv_writer.session_id}/")
    print("  • candles.csv - OHLCV data")
    print("  • trades.csv - Trade history")
    print("  • session_summary.csv - Final metrics (generate via API)")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, use_reloader=False)