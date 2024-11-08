from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from MultibaggerStockAnalyzer import MultibaggerStockAnalyzer
import logging
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_stocks():
    try:
        logger.debug("Received /analyze request")
        
        data = request.get_json()
        if not data or 'investmentAmount' not in data:
            logger.error("Invalid request: missing investmentAmount")
            return jsonify({"error": "Invalid request: missing investmentAmount"}), 400

        investment_amount = data['investmentAmount']
        logger.debug(f"Investment Amount: {investment_amount}")

        try:
            investment_amount = float(investment_amount)
        except ValueError:
            logger.error("Invalid investment amount: not a number")
            return jsonify({"error": "Invalid investment amount: not a number"}), 400

        if investment_amount <= 0:
            logger.error("Invalid investment amount: must be positive")
            return jsonify({"error": "Invalid investment amount: must be positive"}), 400

        analyzer = MultibaggerStockAnalyzer(investment_amount=investment_amount, max_workers=10)
        
        results_df = analyzer.analyze_all_stocks()
        logger.debug(f"Analysis completed. Results shape: {results_df.shape}")

        if results_df.empty:
            logger.warning("No results found")
            return jsonify([])

        # Replace NaN, inf, and -inf values with None before converting to JSON
        results_df = results_df.replace([np.nan, np.inf, -np.inf], None)

        # Convert DataFrame to list of dicts
        results = results_df.to_dict('records')

        # Ensure all values are JSON serializable
        for result in results:
            for key, value in result.items():
                if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    result[key] = str(value)

        return jsonify(results)

    except Exception as e:
        logger.exception(f"Error occurred during analysis: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during analysis"}), 500

if __name__ == '__main__':
    app.run(debug=True)
