# A.I-Assignment

Aave V2 Wallet Credit Scoring Model
This project develops a machine learning model to assign a credit score (0-1000) to Aave V2 wallets based on their historical transaction behavior. The score aims to reflect a wallet's reliability and responsible usage, with higher scores indicating better behavior and lower scores indicating riskier or potentially exploitative patterns.

Challenge Overview
The task was to analyze raw, transaction-level data from the Aave V2 protocol, specifically focusing on actions like Deposit, Borrow, Repay, RedeemUnderlying, and LiquidationCall. Based on this data, a robust machine learning model was developed to assign a credit score to each unique wallet.

Data Source
The model is trained and operates on a sample of user-transactions provided as a JSON file (~87MB). This file contains transaction-level details including userWallet, action, actionData (containing amount), txHash, and timestamp.

Feature Engineering Strategy
From the raw transaction data, the following features were engineered to capture various aspects of a wallet's interaction with the Aave V2 protocol. These features are designed to proxy creditworthiness:

deposit_count / total_value_deposited: Number and total value of assets deposited. Indicates participation and capital provided.

borrow_count / total_value_borrowed: Number and total value of assets borrowed. Indicates reliance on protocol's lending.

repay_count / total_value_repaid: Number and total value of repaid loans. Crucial for creditworthiness.

redeem_count / total_value_redeemed: Number and total value of redeemed collateral. Frequent redemptions might suggest short-term engagement.

liquidation_count / total_value_liquidated: Number and total value of liquidations incurred by the wallet. A strong negative indicator, reflecting failure to maintain collateral.

account_age_days: Duration between the first and last transaction. Longer activity periods might imply stability.

borrow_repay_ratio: total_value_repaid / total_value_borrowed. A ratio significantly greater than 1 suggests over-repayment or consistent good behavior. A ratio near 0 or less indicates unrepaid debt.

net_borrow_exposure: total_value_borrowed - total_value_repaid. Represents the outstanding borrowed amount. Higher values are riskier.

liquidation_rate: liquidation_count / total_transactions. Frequency of liquidations relative to total activity.

Score Logic and Interpretation (0-1000)
Since explicit credit scores were not provided, a heuristic target score was defined to train the supervised learning model. This heuristic assigns a score between 0 and 1000 based on predefined rules that reflect 'good' vs. 'bad' DeFi behavior:

Higher Scores (e.g., 700-1000): Indicative of wallets that consistently repay their loans, have a low (ideally zero) number of liquidations, show substantial value repaid, and demonstrate sustained activity on the platform. These wallets represent reliable and responsible usage.

Mid Scores (e.g., 400-699): May represent wallets with mixed activity, occasional borrowing and repayment, but perhaps some unrepaid exposure or limited history.

Lower Scores (e.g., 0-399): Reflect wallets that have experienced liquidations, have a significant net_borrow_exposure, or exhibit patterns suggestive of risky, bot-like, or exploitative behavior (e.g., many small, rapid transactions that could be part of an exploit, although further analysis would be needed to confirm 'bot-like' specifically).

The Gradient Boosting Regressor then learns to map the engineered features to this heuristic score.

Model Architecture
A Gradient Boosting Regressor from scikit-learn was chosen for this task.

Reasoning: Gradient Boosting models are powerful, robust to various data types, capable of capturing non-linear relationships, and provide insights into feature importance, which aids in explaining the score logic.

Preprocessing: Features are scaled using StandardScaler to ensure all features contribute equally to the model, regardless of their original magnitude.

How to Set Up and Run the Code
Download the Data:
Download the user-transactions.json file from:
https://drive.google.com/file/d/1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS/view?usp=sharing
Rename the downloaded file to user-transactions.json and place it in the root directory of your project.

Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Create and Activate Virtual Environment:

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install Dependencies:

pip install pandas numpy scikit-learn

Run the Jupyter Notebook:

jupyter lab

Open the aave_credit_score.ipynb notebook (or whatever you named it) and run all cells sequentially. This will:

Load the data.

Engineer features.

Define and assign heuristic scores.

Train the ML model and save model.joblib and scaler.joblib in the same directory.

Demonstrate the "one-step" scoring process by loading the saved model and generating scores, saving them to wallet_credit_scores_generated.json.

Output:
The final wallet credit scores will be saved in wallet_credit_scores_generated.json in the same directory.

Extensibility and Future Improvements
Advanced Feature Engineering: Incorporate time-series analysis (e.g., moving averages of debt, repayment streaks), asset-specific features (volatility of collateral assets), and network graph analysis (interactions between wallets).

External Data Integration: Leverage external DeFi data sources (e.g., oracle prices, total value locked, token liquidity) for richer context.

Dynamic Target Definition: Develop a more sophisticated, potentially unsupervised or semi-supervised approach to define "creditworthiness" if labeled data becomes available.

Explainable AI (XAI): Implement techniques like SHAP or LIME to provide more granular explanations for individual wallet scores, beyond just feature importance.

Temporal Validation: Evaluate the model's performance on future data to ensure its robustness over time.

Risk Categorization: Instead of just a score, categorize wallets into risk buckets (e.g., Low Risk, Medium Risk, High Risk).
